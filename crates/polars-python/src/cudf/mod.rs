//! Translate Polars' post-optimization `IR` into cudf-polars's own IR nodes.
//!
//! cudf-polars passes in a "factory" Python object whose methods construct
//! its IR classes.

mod expr;
mod plan;

use polars_core::schema::Schema;
use polars_plan::prelude::{AExpr, ToFieldContext};
use pyo3::IntoPyObject;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::conversion::Wrap;
use crate::lazyframe::visit::NodeTraverser;

/// Translate the plan rooted at `nt` via the given Python `factory`.
#[pyfunction]
fn translate_ir(
    py: Python<'_>,
    nt: PyRef<'_, NodeTraverser>,
    factory: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let (lp_arena, expr_arena) = nt.get_arenas();
    let root = nt.root();
    let lp_arena = lp_arena.lock().unwrap();
    let expr_arena = expr_arena.lock().unwrap();
    let factory = factory.bind(py);
    self::plan::translate_ir(py, root, &lp_arena, &expr_arena, factory)
}

pub(crate) fn build_ir<'py, A>(
    py: Python<'py>,
    factory: &Bound<'py, PyAny>,
    method: &str,
    args: A,
    schema: &Schema,
) -> PyResult<Py<PyAny>>
where
    A: IntoPyObject<'py, Target = PyTuple> + pyo3::call::PyCallArgs<'py>,
{
    match factory.call_method1(method, args) {
        Ok(result) => Ok(result.unbind()),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let msg = format!("{}", e.value(py));
            unsupported_ir(py, factory, method, Some(&msg), schema)
        },
        Err(e) => Err(e),
    }
}

pub(crate) fn build_expr<'py, A>(
    py: Python<'py>,
    factory: &Bound<'py, PyAny>,
    method: &str,
    args: A,
    ae: &AExpr,
    ctx: &ToFieldContext<'_>,
) -> PyResult<Py<PyAny>>
where
    A: IntoPyObject<'py, Target = PyTuple> + pyo3::call::PyCallArgs<'py>,
{
    match factory.call_method1(method, args) {
        Ok(result) => Ok(result.unbind()),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let msg = format!("{}", e.value(py));
            unsupported_expr(py, factory, method, Some(&msg), ae, ctx)
        },
        Err(e) => Err(e),
    }
}

pub(crate) fn unsupported_ir(
    py: Python<'_>,
    factory: &Bound<'_, PyAny>,
    variant_name: &str,
    reason: Option<&str>,
    schema: &Schema,
) -> PyResult<Py<PyAny>> {
    let schema_py = Wrap(schema.clone()).into_py_any(py)?;
    factory
        .call_method1("unsupported_ir", (variant_name, reason, schema_py))
        .map(|b| b.unbind())
}

pub(crate) fn unsupported_expr(
    py: Python<'_>,
    factory: &Bound<'_, PyAny>,
    variant_name: &str,
    reason: Option<&str>,
    ae: &AExpr,
    ctx: &ToFieldContext<'_>,
) -> PyResult<Py<PyAny>> {
    let dtype = match ae.to_field(ctx) {
        Ok(field) => Wrap(field.dtype).into_py_any(py)?,
        Err(_) => py.None(),
    };
    factory
        .call_method1("unsupported_expr", (variant_name, reason, dtype))
        .map(|b| b.unbind())
}

#[pymodule(gil_used = false)]
pub fn cudf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(translate_ir, m)?)?;
    Ok(())
}
