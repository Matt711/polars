//! `AExpr` translation.

use polars_core::prelude::{AnyValue, Schema};
use polars_plan::plans::{DynLiteralValue, LiteralValue, ToFieldContext};
use polars_plan::prelude::{AExpr, Operator};
use polars_utils::arena::{Arena, Node};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::cudf::{build_expr, unsupported_expr};
use crate::series::PySeries;

pub(crate) fn translate_aexpr(
    py: Python<'_>,
    node: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ae = expr_arena.get(node);
    let ctx = ToFieldContext::new(expr_arena, schema);
    match ae {
        AExpr::Column(name) => {
            let dtype = aexpr_dtype(ae, &ctx, py)?;
            build_expr(
                py,
                factory,
                "col",
                (name.as_str(), dtype),
                ae,
                &ctx,
            )
        },
        AExpr::Literal(lit) => translate_literal(py, lit, factory, ae, &ctx),
        AExpr::BinaryExpr { left, op, right } => {
            let dtype = aexpr_dtype(ae, &ctx, py)?;
            let left_py = translate_aexpr(py, *left, expr_arena, schema, factory)?;
            let right_py = translate_aexpr(py, *right, expr_arena, schema, factory)?;
            build_expr(
                py,
                factory,
                "binop",
                (left_py, operator_name(*op), right_py, dtype),
                ae,
                &ctx,
            )
        },
        AExpr::Cast {
            expr,
            dtype,
            options,
        } => {
            let expr_py = translate_aexpr(py, *expr, expr_arena, schema, factory)?;
            let dtype_py = Wrap(dtype.clone()).into_py_any(py)?;
            build_expr(
                py,
                factory,
                "cast",
                (expr_py, dtype_py, options.is_strict()),
                ae,
                &ctx,
            )
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let dtype = aexpr_dtype(ae, &ctx, py)?;
            let predicate_py = translate_aexpr(py, *predicate, expr_arena, schema, factory)?;
            let truthy_py = translate_aexpr(py, *truthy, expr_arena, schema, factory)?;
            let falsy_py = translate_aexpr(py, *falsy, expr_arena, schema, factory)?;
            build_expr(
                py,
                factory,
                "ternary",
                (predicate_py, truthy_py, falsy_py, dtype),
                ae,
                &ctx,
            )
        },

        AExpr::Element => unsupported_expr(py, factory, "AExpr::Element", None, ae, &ctx),
        AExpr::Explode { .. } => unsupported_expr(py, factory, "AExpr::Explode", None, ae, &ctx),
        AExpr::StructField(_) => unsupported_expr(py, factory, "AExpr::StructField", None, ae, &ctx),
        AExpr::Sort { .. } => unsupported_expr(py, factory, "AExpr::Sort", None, ae, &ctx),
        AExpr::Gather { .. } => unsupported_expr(py, factory, "AExpr::Gather", None, ae, &ctx),
        AExpr::SortBy { .. } => unsupported_expr(py, factory, "AExpr::SortBy", None, ae, &ctx),
        AExpr::Filter { .. } => unsupported_expr(py, factory, "AExpr::Filter", None, ae, &ctx),
        AExpr::Agg(_) => unsupported_expr(py, factory, "AExpr::Agg", None, ae, &ctx),
        AExpr::AnonymousAgg { .. } => unsupported_expr(py, factory,
            "AExpr::AnonymousAgg", Some("anonymous aggregates cannot be translated"), ae, &ctx),
        AExpr::AnonymousFunction { .. } => unsupported_expr(py, factory,
            "AExpr::AnonymousFunction", Some("anonymous functions cannot be translated"), ae, &ctx),
        AExpr::Eval { .. } => unsupported_expr(py, factory, "AExpr::Eval", None, ae, &ctx),
        AExpr::StructEval { .. } => unsupported_expr(py, factory, "AExpr::StructEval", None, ae, &ctx),
        AExpr::Function { .. } => unsupported_expr(py, factory, "AExpr::Function", None, ae, &ctx),
        AExpr::Over { .. } => unsupported_expr(py, factory, "AExpr::Over", None, ae, &ctx),
        AExpr::Rolling { .. } => unsupported_expr(py, factory, "AExpr::Rolling", None, ae, &ctx),
        AExpr::Slice { .. } => unsupported_expr(py, factory, "AExpr::Slice", None, ae, &ctx),
        AExpr::Len => unsupported_expr(py, factory, "AExpr::Len", None, ae, &ctx),
    }
}

fn aexpr_dtype(
    ae: &AExpr,
    ctx: &ToFieldContext<'_>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let field = ae
        .to_field(ctx)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Wrap(field.dtype).into_py_any(py)
}

fn translate_literal(
    py: Python<'_>,
    lit: &LiteralValue,
    factory: &Bound<'_, PyAny>,
    ae: &AExpr,
    ctx: &ToFieldContext<'_>,
) -> PyResult<Py<PyAny>> {
    let dtype = Wrap(lit.get_datatype()).into_py_any(py)?;
    let value = match lit {
        LiteralValue::Dyn(d) => match d {
            DynLiteralValue::Int(v) => v.into_py_any(py)?,
            DynLiteralValue::Float(v) => v.into_py_any(py)?,
            DynLiteralValue::Str(v) => v.into_py_any(py)?,
            DynLiteralValue::List(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "list literal",
                ));
            },
        },
        LiteralValue::Scalar(sc) => match sc.as_any_value() {
            AnyValue::Duration(delta, _) => delta.into_py_any(py)?,
            any => Wrap(any).into_py_any(py)?,
        },
        LiteralValue::Range(_) => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "range literal",
            ));
        },
        LiteralValue::Series(s) => PySeries::from((**s).clone()).into_py_any(py)?,
    };
    build_expr(py, factory, "literal", (value, dtype), ae, ctx)
}

fn operator_name(op: Operator) -> &'static str {
    match op {
        Operator::Eq => "Eq",
        Operator::EqValidity => "EqValidity",
        Operator::NotEq => "NotEq",
        Operator::NotEqValidity => "NotEqValidity",
        Operator::Lt => "Lt",
        Operator::LtEq => "LtEq",
        Operator::Gt => "Gt",
        Operator::GtEq => "GtEq",
        Operator::Plus => "Plus",
        Operator::Minus => "Minus",
        Operator::Multiply => "Multiply",
        Operator::RustDivide => "RustDivide",
        Operator::TrueDivide => "TrueDivide",
        Operator::FloorDivide => "FloorDivide",
        Operator::Modulus => "Modulus",
        Operator::And => "And",
        Operator::Or => "Or",
        Operator::Xor => "Xor",
        Operator::LogicalAnd => "LogicalAnd",
        Operator::LogicalOr => "LogicalOr",
    }
}
