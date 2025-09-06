use gpu_programming::run;

fn test(){
     let x: Vec<u8> = Vec::from([1,2,3,4,5,6]);
     let data: &[u16] = bytemuck::cast_slice(&x); 
     println!("{:#?}",data);
 }

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    test();
    run();
    Ok(())
}
