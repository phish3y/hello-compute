struct Point {
    x: f32,
    y: f32,
};

@group(0) @binding(0)
var<storage, read> points_a: array<Point>;

@group(0) @binding(1)
var<storage, read> points_b: array<Point>;

@group(0) @binding(2)
var<storage, read_write> distances: array<f32>;

fn radians(degrees: f32) -> f32 {
    return degrees * 0.01745329252; // PI / 180
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&points_a)) {
        return;
    }

    let R = 6371008.7714; // Radius of the Earth in meters

    let y1 = radians(points_a[i].y);
    let x1 = radians(points_a[i].x);
    let y2 = radians(points_b[i].y);
    let x2 = radians(points_b[i].x);

    let dy = y2 - y1;
    let dx = x2 - x1;

    let a = sin(dy / 2.0) * sin(dy / 2.0) +
        cos(y1) * cos(y2) *
        sin(dx / 2.0) * sin(dx / 2.0);


    let c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

    distances[i] = R * c;
}
