//! `IR` translation.

use std::sync::Arc;

use polars_core::schema::{Schema, SchemaRef};
use polars_plan::dsl::JoinOptionsIR;
use polars_plan::plans::{FileInfo, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::{
    AExpr, FileScanIR, IRFunctionExpr, ScanSources, UnifiedScanArgs,
};
use polars_utils::IdxSize;
use polars_utils::arena::{Arena, Node};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::conversion::Wrap;
use crate::cudf::expr::translate_aexpr;
use crate::cudf::{build_ir, unsupported_ir};
use crate::lazyframe::visitor::nodes::serialize_scan_type;

pub(crate) fn translate_ir(
    py: Python<'_>,
    node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ir = lp_arena.get(node);
    match ir {
        IR::Scan {
            sources,
            file_info,
            hive_parts: _,
            predicate,
            predicate_file_skip_applied: _,
            output_schema,
            scan_type,
            unified_scan_args,
        } => translate_scan(
            py,
            sources,
            file_info,
            output_schema.as_ref(),
            predicate.as_ref(),
            scan_type,
            unified_scan_args,
            expr_arena,
            factory,
        ),
        IR::Filter { input, predicate } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let translated_input =
                translate_ir(py, *input, lp_arena, expr_arena, factory)?;
            // Drop predicate-pushdown's `DynamicPred` marker.
            if is_dynamic_predicate(predicate.node(), expr_arena) {
                return Ok(translated_input);
            }
            let translated_predicate =
                translate_named_expr(py, predicate, expr_arena, &input_schema, factory)?;
            let schema = Wrap((**input_schema).clone()).into_py_any(py)?;
            build_ir(
                py,
                factory,
                "filter",
                (translated_input, translated_predicate, schema),
                &input_schema,
            )
        },
        IR::SimpleProjection { input, columns } => {
            let translated_input =
                translate_ir(py, *input, lp_arena, expr_arena, factory)?;
            let schema = Wrap((**columns).clone()).into_py_any(py)?;
            build_ir(
                py,
                factory,
                "simple_projection",
                (translated_input, schema),
                columns,
            )
        },
        IR::Select {
            input,
            expr,
            schema,
            options,
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let translated_input =
                translate_ir(py, *input, lp_arena, expr_arena, factory)?;
            let exprs = translate_expr_ir_list(py, expr, expr_arena, &input_schema, factory)?;
            let schema_py = Wrap((**schema).clone()).into_py_any(py)?;
            build_ir(
                py,
                factory,
                "select",
                (translated_input, exprs, schema_py, options.should_broadcast),
                schema,
            )
        },
        IR::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            schema,
            options,
        } => translate_join(
            py,
            *input_left,
            *input_right,
            left_on,
            right_on,
            schema,
            options,
            lp_arena,
            expr_arena,
            factory,
        ),

        IR::PythonScan { options: _ } => unsupported_ir(py, factory,
            "IR::PythonScan", Some("Python data sources are unsupported"), &ir.schema(lp_arena)),
        IR::Slice { input: _, offset: _, len: _ } => unsupported_ir(py, factory,
            "IR::Slice", None, &ir.schema(lp_arena)),
        IR::DataFrameScan { df: _, schema: _, output_schema: _ } => unsupported_ir(py, factory,
            "IR::DataFrameScan", None, &ir.schema(lp_arena)),
        IR::Sort { input: _, by_column: _, slice: _, sort_options: _ } => unsupported_ir(py, factory,
            "IR::Sort", None, &ir.schema(lp_arena)),
        IR::Cache { input: _, id: _ } => unsupported_ir(py, factory,
            "IR::Cache", None, &ir.schema(lp_arena)),
        IR::GroupBy { input: _, keys: _, aggs: _, schema: _, maintain_order: _, options: _, apply: _ } =>
            unsupported_ir(py, factory, "IR::GroupBy", None, &ir.schema(lp_arena)),
        IR::Gather { input: _, idxs: _, null_on_oob: _ } => unsupported_ir(py, factory,
            "IR::Gather", None, &ir.schema(lp_arena)),
        IR::HStack { input: _, exprs: _, schema: _, options: _ } => unsupported_ir(py, factory,
            "IR::HStack", None, &ir.schema(lp_arena)),
        IR::Distinct { input: _, options: _ } => unsupported_ir(py, factory,
            "IR::Distinct", None, &ir.schema(lp_arena)),
        IR::MapFunction { input: _, function: _ } => unsupported_ir(py, factory,
            "IR::MapFunction", None, &ir.schema(lp_arena)),
        IR::Union { inputs: _, options: _ } => unsupported_ir(py, factory,
            "IR::Union", None, &ir.schema(lp_arena)),
        IR::HConcat { inputs: _, schema: _, options: _ } => unsupported_ir(py, factory,
            "IR::HConcat", None, &ir.schema(lp_arena)),
        IR::ExtContext { input: _, contexts: _, schema: _ } => unsupported_ir(py, factory,
            "IR::ExtContext", None, &ir.schema(lp_arena)),
        IR::Sink { input: _, payload: _ } => unsupported_ir(py, factory,
            "IR::Sink", None, &ir.schema(lp_arena)),
        IR::SinkMultiple { inputs: _ } => unsupported_ir(py, factory,
            "IR::SinkMultiple", None, &ir.schema(lp_arena)),
        IR::MergeSorted { input_left: _, input_right: _, key: _, maintain_order: _ } =>
            unsupported_ir(py, factory, "IR::MergeSorted", None, &ir.schema(lp_arena)),
        IR::UnoptimizedDispatch { inputs: _, arg_map: _, operation: _ } => unsupported_ir(py, factory,
            "IR::UnoptimizedDispatch", None, &ir.schema(lp_arena)),
        IR::Invalid => unsupported_ir(py, factory,
            "IR::Invalid", Some("invalid IR node reached translator"), &Schema::default()),
    }
}

#[allow(clippy::too_many_arguments)]
fn translate_join(
    py: Python<'_>,
    input_left: Node,
    input_right: Node,
    left_on: &[ExprIR],
    right_on: &[ExprIR],
    schema: &SchemaRef,
    options: &Arc<JoinOptionsIR>,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use polars_ops::frame::JoinType;
    use polars_plan::dsl::JoinTypeOptionsIR;

    let left_schema = lp_arena.get(input_left).schema(lp_arena);
    let right_schema = lp_arena.get(input_right).schema(lp_arena);
    let translated_left = translate_ir(py, input_left, lp_arena, expr_arena, factory)?;
    let translated_right = translate_ir(py, input_right, lp_arena, expr_arena, factory)?;
    let translated_left_on =
        translate_expr_ir_list(py, left_on, expr_arena, &left_schema, factory)?;
    let translated_right_on =
        translate_expr_ir_list(py, right_on, expr_arena, &right_schema, factory)?;
    let schema_py = Wrap((**schema).clone()).into_py_any(py)?;
    let suffix = options
        .args
        .suffix
        .as_ref()
        .map(|s| s.as_str().to_owned())
        .unwrap_or_else(|| "_right".to_owned());

    let how = match &options.args.how {
        JoinType::Inner => "Inner",
        JoinType::Left => "Left",
        JoinType::Right => "Right",
        JoinType::Full => "Full",
        JoinType::Cross => "Cross",
        JoinType::Semi => "Semi",
        JoinType::Anti => "Anti",
        JoinType::AsOf(_) => {
            return unsupported_ir(py, factory, "IR::Join (AsOf)", None, schema);
        },
        JoinType::IEJoin => {
            let Some(JoinTypeOptionsIR::IEJoin(ie_opts)) = options.options.as_ref()
            else {
                unreachable!(
                    "JoinType::IEJoin without an IEJoinOptions payload"
                );
            };
            let (op1, op2) = (ie_opts.operator1, ie_opts.operator2);
            let op1_name = inequality_op_name(op1);
            let op2_name = op2.map(inequality_op_name);
            return build_ir(
                py,
                factory,
                "conditional_join",
                (
                    translated_left,
                    translated_right,
                    translated_left_on,
                    translated_right_on,
                    schema_py,
                    op1_name,
                    op2_name,
                    options.args.nulls_equal,
                    suffix,
                    options.args.should_coalesce(),
                ),
                schema,
            );
        },
        JoinType::Range => {
            return unsupported_ir(py, factory, "IR::Join (Range)", None, schema);
        },
    };

    build_ir(
        py,
        factory,
        "join",
        (
            translated_left,
            translated_right,
            translated_left_on,
            translated_right_on,
            schema_py,
            how,
            options.args.nulls_equal,
            suffix,
            options.args.should_coalesce(),
        ),
        schema,
    )
}

fn inequality_op_name(op: polars_ops::frame::InequalityOperator) -> &'static str {
    use polars_ops::frame::InequalityOperator;
    match op {
        InequalityOperator::Lt => "Lt",
        InequalityOperator::LtEq => "LtEq",
        InequalityOperator::Gt => "Gt",
        InequalityOperator::GtEq => "GtEq",
    }
}

#[allow(clippy::too_many_arguments)]
fn translate_scan(
    py: Python<'_>,
    sources: &ScanSources,
    file_info: &FileInfo,
    output_schema: Option<&SchemaRef>,
    predicate: Option<&ExprIR>,
    scan_type: &FileScanIR,
    unified_scan_args: &UnifiedScanArgs,
    expr_arena: &Arena<AExpr>,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let Some((typ, reader_options)) = serialize_scan_type(scan_type)? else {
        return unsupported_ir(py, factory,
            "IR::Scan", Some("unsupported scan source"), &file_info.schema);
    };

    let paths: Vec<String> = match sources {
        ScanSources::Paths(paths) => paths
            .as_ref()
            .iter()
            .map(|p| p.as_std_path().to_string_lossy().into_owned())
            .collect(),
        ScanSources::Files(_) | ScanSources::Buffers(_) => {
            return unsupported_ir(py, factory,
                "IR::Scan",
                Some("only path-based scan sources are translated"),
                &file_info.schema);
        },
    };

    if paths.is_empty() {
        let schema_py = Wrap((*file_info.schema).clone()).into_py_any(py)?;
        return build_ir(py, factory, "empty", (schema_py,), &file_info.schema);
    }

    if unified_scan_args.deletion_files.is_some() {
        return unsupported_ir(
            py,
            factory,
            "IR::Scan",
            Some(
                "Iceberg format is not supported in cudf-polars. \
                 Furthermore, row-level deletions are not supported.",
            ),
            &file_info.schema,
        );
    }

    // Reject compressed csv / ndjson; libcudf reads them uncompressed only.
    if matches!(typ, "csv" | "ndjson") {
        for path in &paths {
            if path.contains("://") && !path.starts_with("file://") {
                continue;
            }
            let local = path.strip_prefix("file://").unwrap_or(path);
            let mut buf = [0u8; 4];
            let read_result = std::fs::File::open(local).and_then(|mut f| {
                use std::io::Read;
                f.read(&mut buf)
            });
            let Ok(n) = read_result else {
                continue;
            };
            if polars_io::utils::compression::SupportedCompression::check(&buf[..n])
                .is_some()
            {
                let msg = format!(
                    "Reading compressed {} files is not supported.",
                    typ.to_uppercase()
                );
                return unsupported_ir(py, factory, "IR::Scan", Some(&msg), &file_info.schema);
            }
        }
    }

    let cloud_options = unified_scan_args
        .cloud_options
        .as_ref()
        .map(|co| {
            serde_json::to_string(co)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(format!("{err:?}")))
        })
        .transpose()?;

    let with_columns: Option<Vec<String>> = unified_scan_args
        .projection
        .as_ref()
        .map(|p| p.iter().map(|s| s.as_str().to_owned()).collect());

    let (skip_rows, n_rows) = match unified_scan_args.pre_slice.as_ref() {
        None => (0i64, -1i64),
        Some(slice) => {
            let (offset, len) = slice.to_signed_offset_len();
            let n = if len == IdxSize::MAX { -1i64 } else { len as i64 };
            (offset, n)
        },
    };

    let row_index = unified_scan_args
        .row_index
        .as_ref()
        .map(|ri| (ri.name.as_str().to_owned(), ri.offset));

    let include_file_paths = unified_scan_args
        .include_file_paths
        .as_ref()
        .map(|s| s.as_str().to_owned());

    let translated_predicate = match predicate {
        Some(p) if !is_dynamic_predicate(p.node(), expr_arena) => {
            Some(translate_named_expr(py, p, expr_arena, &file_info.schema, factory)?)
        },
        _ => None,
    };
    let schema_ref = output_schema.unwrap_or(&file_info.schema);
    let schema = Wrap((**schema_ref).clone()).into_py_any(py)?;
    let paths_py = PyList::new(py, paths)?.into_any().unbind();

    build_ir(
        py,
        factory,
        "scan",
        (
            paths_py,
            schema,
            translated_predicate,
            typ,
            reader_options,
            cloud_options,
            with_columns,
            skip_rows,
            n_rows,
            row_index,
            include_file_paths,
        ),
        schema_ref,
    )
}

pub(crate) fn translate_named_expr(
    py: Python<'_>,
    e: &ExprIR,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let translated = translate_aexpr(py, e.node(), expr_arena, schema, factory)?;
    let output_name = e.output_name().as_str().to_owned();
    factory
        .call_method1("named_expr", (translated, output_name))
        .map(|b| b.unbind())
}

fn is_dynamic_predicate(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    matches!(
        expr_arena.get(node),
        AExpr::Function {
            function: IRFunctionExpr::DynamicPred { .. },
            ..
        }
    )
}

fn translate_expr_ir_list(
    py: Python<'_>,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let items: Vec<Py<PyAny>> = exprs
        .iter()
        .map(|e| translate_named_expr(py, e, expr_arena, schema, factory))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, items)?.into_any().unbind())
}
