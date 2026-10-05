use ort::session::Session;

fn main() -> ort::Result<()> {
    let path = std::env::args().nth(1).expect("usage: probe <model.onnx>");
    let session = Session::builder()?.commit_from_file(path)?;
    for i in session.inputs() {
        println!("in  {} {:?}", i.name(), i.dtype());
    }
    for o in session.outputs() {
        println!("out {} {:?}", o.name(), o.dtype());
    }
    let meta = session.metadata()?;
    for key in meta.custom_keys()? {
        println!("meta {key} = {:?}", meta.custom(&key));
    }
    Ok(())
}
