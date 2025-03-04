use std::fmt;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use polars_utils::arena::Node;
#[cfg(feature = "serde")]
use polars_utils::pl_serialize;
use recursive::recursive;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;
// (Major, Minor)
// Add a field -> increment minor
// Remove or modify a field -> increment major and reset minor
pub static DSL_VERSION: (u16, u16) = (0, 1);
static DSL_MAGIC_BYTES: &[u8] = b"DSL_VERSION";

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DslPlan {
    #[cfg(feature = "python")]
    PythonScan {
        options: crate::dsl::python_dsl::PythonOptionsDsl,
    },
    /// Filter on a boolean mask
    Filter {
        input: Arc<DslPlan>,
        predicate: Expr,
    },
    /// Cache the input at this point in the LP
    Cache { input: Arc<DslPlan>, id: usize },
    Scan {
        sources: ScanSources,
        /// Materialized at IR except for AnonymousScan.
        file_info: Option<FileInfo>,
        file_options: FileScanOptions,
        scan_type: FileScan,
        /// Local use cases often repeatedly collect the same `LazyFrame` (e.g. in interactive notebook use-cases),
        /// so we cache the IR conversion here, as the path expansion can be quite slow (especially for cloud paths).
        #[cfg_attr(feature = "serde", serde(skip))]
        cached_ir: Arc<Mutex<Option<IR>>>,
    },
    // we keep track of the projection and selection as it is cheaper to first project and then filter
    /// In memory DataFrame
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
    },
    /// Polars' `select` operation, this can mean projection, but also full data access.
    Select {
        expr: Vec<Expr>,
        input: Arc<DslPlan>,
        options: ProjectionOptions,
    },
    /// Groupby aggregation
    GroupBy {
        input: Arc<DslPlan>,
        keys: Vec<Expr>,
        aggs: Vec<Expr>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
        #[cfg_attr(feature = "serde", serde(skip))]
        apply: Option<(Arc<dyn DataFrameUdf>, SchemaRef)>,
    },
    /// Join operation
    Join {
        input_left: Arc<DslPlan>,
        input_right: Arc<DslPlan>,
        // Invariant: left_on and right_on are equal length.
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        // Invariant: Either left_on/right_on or predicates is set (non-empty).
        predicates: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Arc<DslPlan>,
        exprs: Vec<Expr>,
        options: ProjectionOptions,
    },
    /// Remove duplicates from the table
    Distinct {
        input: Arc<DslPlan>,
        options: DistinctOptionsDSL,
    },
    /// Sort the table
    Sort {
        input: Arc<DslPlan>,
        by_column: Vec<Expr>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    /// Slice the table
    Slice {
        input: Arc<DslPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A (User Defined) Function
    MapFunction {
        input: Arc<DslPlan>,
        function: DslFunction,
    },
    /// Vertical concatenation
    Union {
        inputs: Vec<DslPlan>,
        args: UnionArgs,
    },
    /// Horizontal concatenation of multiple plans
    HConcat {
        inputs: Vec<DslPlan>,
        options: HConcatOptions,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Arc<DslPlan>,
        contexts: Vec<DslPlan>,
    },
    Sink {
        input: Arc<DslPlan>,
        payload: SinkType,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: Arc<DslPlan>,
        input_right: Arc<DslPlan>,
        key: PlSmallStr,
    },
    IR {
        // Keep the original Dsl around as we need that for serialization.
        dsl: Arc<DslPlan>,
        version: u32,
        #[cfg_attr(feature = "serde", serde(skip))]
        node: Option<Node>,
    },
}

impl Clone for DslPlan {
    // Autogenerated by rust-analyzer, don't care about it looking nice, it just
    // calls clone on every member of every enum variant.
    #[rustfmt::skip]
    #[allow(clippy::clone_on_copy)]
    #[recursive]
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "python")]
            Self::PythonScan { options } => Self::PythonScan { options: options.clone() },
            Self::Filter { input, predicate } => Self::Filter { input: input.clone(), predicate: predicate.clone() },
            Self::Cache { input, id } => Self::Cache { input: input.clone(), id: id.clone() },
            Self::Scan { sources, file_info, file_options, scan_type, cached_ir } => Self::Scan { sources: sources.clone(), file_info: file_info.clone(), file_options: file_options.clone(), scan_type: scan_type.clone(), cached_ir: cached_ir.clone() },
            Self::DataFrameScan { df, schema, } => Self::DataFrameScan { df: df.clone(), schema: schema.clone(),  },
            Self::Select { expr, input, options } => Self::Select { expr: expr.clone(), input: input.clone(), options: options.clone() },
            Self::GroupBy { input, keys, aggs,  apply, maintain_order, options } => Self::GroupBy { input: input.clone(), keys: keys.clone(), aggs: aggs.clone(), apply: apply.clone(), maintain_order: maintain_order.clone(), options: options.clone() },
            Self::Join { input_left, input_right, left_on, right_on, predicates, options } => Self::Join { input_left: input_left.clone(), input_right: input_right.clone(), left_on: left_on.clone(), right_on: right_on.clone(), options: options.clone(), predicates: predicates.clone() },
            Self::HStack { input, exprs, options } => Self::HStack { input: input.clone(), exprs: exprs.clone(),  options: options.clone() },
            Self::Distinct { input, options } => Self::Distinct { input: input.clone(), options: options.clone() },
            Self::Sort {input,by_column, slice, sort_options } => Self::Sort { input: input.clone(), by_column: by_column.clone(), slice: slice.clone(), sort_options: sort_options.clone() },
            Self::Slice { input, offset, len } => Self::Slice { input: input.clone(), offset: offset.clone(), len: len.clone() },
            Self::MapFunction { input, function } => Self::MapFunction { input: input.clone(), function: function.clone() },
            Self::Union { inputs, args} => Self::Union { inputs: inputs.clone(), args: args.clone() },
            Self::HConcat { inputs, options } => Self::HConcat { inputs: inputs.clone(), options: options.clone() },
            Self::ExtContext { input, contexts, } => Self::ExtContext { input: input.clone(), contexts: contexts.clone() },
            Self::Sink { input, payload } => Self::Sink { input: input.clone(), payload: payload.clone() },
            #[cfg(feature = "merge_sorted")]
            Self::MergeSorted { input_left, input_right, key } => Self::MergeSorted { input_left: input_left.clone(), input_right: input_right.clone(), key: key.clone() },
            Self::IR {node, dsl, version} => Self::IR {node: *node, dsl: dsl.clone(), version: *version},
        }
    }
}

impl Default for DslPlan {
    fn default() -> Self {
        let df = DataFrame::empty();
        let schema = df.schema().clone();
        DslPlan::DataFrameScan {
            df: Arc::new(df),
            schema,
        }
    }
}

impl DslPlan {
    pub fn describe(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe())
    }

    pub fn describe_tree_format(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe_tree_format())
    }

    pub fn display(&self) -> PolarsResult<impl fmt::Display> {
        struct DslPlanDisplay(IRPlan);
        impl fmt::Display for DslPlanDisplay {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0.as_ref().display(), f)
            }
        }
        Ok(DslPlanDisplay(self.clone().to_alp()?))
    }

    pub fn to_alp(self) -> PolarsResult<IRPlan> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);

        let node = to_alp(
            self,
            &mut expr_arena,
            &mut lp_arena,
            &mut OptFlags::default(),
        )?;
        let plan = IRPlan::new(node, lp_arena, expr_arena);

        Ok(plan)
    }

    #[cfg(feature = "serde")]
    pub fn serialize_versioned<W: Write>(&self, mut writer: W) -> PolarsResult<()> {
        let le_major = DSL_VERSION.0.to_le_bytes();
        let le_minor = DSL_VERSION.1.to_le_bytes();
        writer.write_all(DSL_MAGIC_BYTES)?;
        writer.write_all(&le_major)?;
        writer.write_all(&le_minor)?;
        pl_serialize::SerializeOptions::default().serialize_into_writer::<_, _, true>(writer, self)
    }

    #[cfg(feature = "serde")]
    pub fn deserialize_versioned<R: Read>(mut reader: R) -> PolarsResult<Self> {
        const MAGIC_LEN: usize = DSL_MAGIC_BYTES.len();
        let mut version_magic = [0u8; MAGIC_LEN + 4];
        reader.read_exact(&mut version_magic)?;

        if &version_magic[..MAGIC_LEN] != DSL_MAGIC_BYTES {
            polars_bail!(ComputeError: "dsl magic bytes not found")
        }

        // The DSL serialization is forward compatible if fields don't change,
        // so we don't check equality here, we just use this version
        // to inform users when the deserialization fails.
        let major = u16::from_be_bytes(version_magic[MAGIC_LEN..MAGIC_LEN + 2].try_into().unwrap());
        let minor = u16::from_be_bytes(
            version_magic[MAGIC_LEN + 2..MAGIC_LEN + 4]
                .try_into()
                .unwrap(),
        );

        pl_serialize::SerializeOptions::default()
                    .deserialize_from_reader::<_, _, true>(reader).map_err(|e| {
                    polars_err!(ComputeError: "deserialization failed\n\ngiven DSL_VERSION: {:?} is not compatible with this Polars version which uses DSL_VERSION: {:?}\nerror: {}", (major, minor), DSL_VERSION, e)
                })
    }
}
