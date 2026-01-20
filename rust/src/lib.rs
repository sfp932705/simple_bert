use pyo3::prelude::*;
pub mod token_encoders;

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let token_encoders_mod = PyModule::new(m.py(), "token_encoders")?;
    token_encoders_mod.add_class::<token_encoders::bpe::RustBPETokenizer>()?;
    token_encoders_mod.add_class::<token_encoders::wp::RustWordPieceTokenizer>()?;
    m.add_submodule(&token_encoders_mod)?;

    Ok(())
}
