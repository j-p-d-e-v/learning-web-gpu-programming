



#[derive(Debug)]
enum Shape{
    Circle { radius: f64 },
    Square { side: f64 },
    Rectangle { width: f64, height: f64 }
}

#[derive(Debug)]
pub struct CircleRadius {
    radius: f64
}

#[derive(Debug)]
enum Shape2{
    Circle(CircleRadius),
    Square { side: f64 },
    Rectangle { width: f64, height: f64 }
}

fn main(){
    let s1: Shape = Shape::Circle {
        radius: 1.3
    };
    let s2: Shape = Shape::Circle {
        radius: 1.3
    };
    println!("1 {:#?}",s1);
    println!("2 {:#?}",s2);
}