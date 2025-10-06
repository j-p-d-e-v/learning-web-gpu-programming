use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;

fn main() -> anyhow::Result<()> {

    //To Study
    println!(r"cargo:return-if-changed=src/res/*");

    let out_dir: String = std::env::var("OUT_DIR").unwrap_or("storage".to_string());
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let mut paths_to_copy = Vec::new();
    paths_to_copy.push(r"src/res/");
    copy_items(&paths_to_copy, out_dir, &copy_options)?;
    Ok(())
}