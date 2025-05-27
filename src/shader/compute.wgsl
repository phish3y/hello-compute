struct Point {
    x: f32,
    y: f32,
    value: f32,
};

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < arrayLength(&points)) {
        points[i].value = points[i].value * 2.0;
    }
}