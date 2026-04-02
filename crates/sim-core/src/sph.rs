use std::f32::consts::PI;

pub const EPSILON: f32 = 1.0e-6;
pub const PREDICTION_FACTOR: f32 = 1.0 / 120.0;

pub const NEIGHBOUR_OFFSETS: [(i32, i32); 9] = [
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

// ── Vec2 ──────────────────────────────────────────────────
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        if rhs.abs() <= EPSILON {
            Self::ZERO
        } else {
            Self::new(self.x / rhs, self.y / rhs)
        }
    }
}

// ── Vec3 ──────────────────────────────────────────────────
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len <= EPSILON {
            Self::ZERO
        } else {
            self / len
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        if rhs.abs() <= EPSILON {
            Self::ZERO
        } else {
            Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
        }
    }
}

// ── SPH Kernels ───────────────────────────────────────────

pub fn smoothing_kernel_poly6(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let v = radius * radius - distance * distance;
        v * v * v * (4.0 / (PI * radius.powi(8)))
    } else {
        0.0
    }
}

pub fn spiky_kernel_pow2(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let v = radius - distance;
        v * v * (6.0 / (PI * radius.powi(4)))
    } else {
        0.0
    }
}

pub fn spiky_kernel_pow3(distance: f32, radius: f32) -> f32 {
    if distance < radius {
        let v = radius - distance;
        v * v * v * (10.0 / (PI * radius.powi(5)))
    } else {
        0.0
    }
}

pub fn derivative_spiky_pow2(distance: f32, radius: f32) -> f32 {
    if distance <= radius {
        let v = radius - distance;
        -v * (12.0 / (PI * radius.powi(4)))
    } else {
        0.0
    }
}

pub fn derivative_spiky_pow3(distance: f32, radius: f32) -> f32 {
    if distance <= radius {
        let v = radius - distance;
        -v * v * (30.0 / (PI * radius.powi(5)))
    } else {
        0.0
    }
}

// ── Spatial Hash ──────────────────────────────────────────

pub fn get_cell(position: Vec2, radius: f32) -> (i32, i32) {
    (
        (position.x / radius).floor() as i32,
        (position.y / radius).floor() as i32,
    )
}

pub fn hash_cell(cell: (i32, i32)) -> u32 {
    let x = cell.0 as u32;
    let y = cell.1 as u32;
    x.wrapping_mul(15_823)
        .wrapping_add(y.wrapping_mul(9_737_333))
}

pub fn signed_unit(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ── Simple RNG ────────────────────────────────────────────

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.state >> 40) as u32;
        bits as f32 / ((1 << 24) - 1) as f32
    }
}
