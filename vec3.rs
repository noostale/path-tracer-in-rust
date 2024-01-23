use crate::rand::Rng;

// VEC3 STRUCTURE

// Struct for a vector of 3 elements
#[derive(Debug, Copy, Clone)]
pub struct Vec3 {
    // Point e defined as [T, N] (Type, Number of elements)
    pub e: [f64; 3],
}

/// Methods for the Vec3 struct
impl Vec3 {
    /// Constructor for the Vec3 structure
    pub fn new(e1: f64, e2: f64, e3: f64) -> Self {
        return Self { e: [e1, e2, e3] };
    }

    /// Getter for x coordinate
    pub fn x(&self) -> f64 {
        return self.e[0];
    }

    /// Getter for y coordinate
    pub fn y(&self) -> f64 {
        return self.e[1];
    }

    /// Getter for z coordinate
    pub fn z(&self) -> f64 {
        return self.e[2];
    }

    pub fn length_squared(&self) -> f64 {
        return self.e[0] * self.e[0] + self.e[1] * self.e[1] + self.e[2] * self.e[2];
    }

    /// Calculate the length of the Vec3
    fn length(&self) -> f64 {
        return self.length_squared().sqrt();
    }

    /// Random vector generator
    fn random() -> Vec3 {
        return Vec3 {
            e: [random_double(), random_double(), random_double()],
        };
    }

    /// Random vector generator in a selected range
    fn random_range(min: f64, max: f64) -> Vec3 {
        return Vec3 {
            e: [
                random_double_range(min, max),
                random_double_range(min, max),
                random_double_range(min, max),
            ],
        };
    }

    /// Random vector generator in a unit sphere
    fn random_in_unit_sphere() -> Vec3 {
        loop {
            let p: Vec3 = Vec3::random_range(-1.0, 1.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    /// Random normalized vector generator in a unit sphere
    pub fn random_unit_vector() -> Vec3 {
        return unit_vector(Vec3::random_in_unit_sphere());
    }

    /// Random vector generator in a unit disk
    fn random_on_hemisphere(normal: Vec3) -> Vec3 {
        let on_unit_sphere: Vec3 = Vec3::random_unit_vector();
        if on_unit_sphere.dot(&normal) > 0.0 {
            on_unit_sphere
        } else {
            let positive_vec: Vec3 = -on_unit_sphere;
            positive_vec
        }
    }

    /// Check if the Vec3 is near zero
    pub fn near_zero(&self) -> bool {
        let s: f64 = 1e-8;
        (self.e[0].abs() < s) && (self.e[1].abs() < s) && (self.e[2].abs() < s)
    }

    /// Calculate the dot product between this Vec3 and another Vec3
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.e[0] * other.e[0] + self.e[1] * other.e[1] + self.e[2] * other.e[2]
    }

    pub fn zero() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }
    
}



/// Function to calculate the cross (vectorial) product between two Vec3
pub fn cross(u: Vec3, v: Vec3) -> Vec3 {
    Vec3::new(
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0],
    )
}

/// Function to calculate the unit vector (normalized vector) of a Vec3
pub fn unit_vector(u: Vec3) -> Vec3 {
    let length: f64 = u.length();
    return u / length;
}

/// Type aliases for Vec3
pub type Point3 = Vec3; // 3D point
pub type Color = Vec3; // RGB color

// RANDOM NUMBER GENERATOR

/// Function to generate a random number between 0 and 1
fn random_double() -> f64 {
    // Returns a random real in [0,1).
    rand::thread_rng().gen_range(0.0..1.0)
}

/// Function to generate a random number between min and max
fn random_double_range(min: f64, max: f64) -> f64 {
    // Returns a random real in [min,max).
    rand::thread_rng().gen_range(min..max)
}

// OPERATOR OVERRIDES

/// Operator ovverride for -Vec3
impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            e: [-self.e[0], -self.e[1], -self.e[2]],
        }
    }
}

/// Operator ovverride for Vec3[]
impl std::ops::Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.e[i]
    }
}

/// Operator ovverride for Vec3+=Vec3
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
        self.e[2] += other.e[2];
    }
}

/// Operator ovverride for Vec3-=Vec3
impl std::ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self.e[0] *= t;
        self.e[1] *= t;
        self.e[2] *= t;
    }
}

/// Operator ovverride for Vec3/=Vec3
impl std::ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        self.e[0] /= 1.0 / t;
        self.e[1] /= 1.0 / t;
        self.e[2] /= 1.0 / t;
    }
}

/// Operator ovverride for Vec3+Vec3
impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] + other.e[0],
            self.e[1] + other.e[1],
            self.e[2] + other.e[2],
        )
    }
}

/// Operator ovverride for Vec3-Vec3
impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] - other.e[0],
            self.e[1] - other.e[1],
            self.e[2] - other.e[2],
        )
    }
}

/// Operator ovverride for Vec3*Vec3
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] * other.e[0],
            self.e[1] * other.e[1],
            self.e[2] * other.e[2],
        )
    }
}

/// Operator ovverride for Vec3*t
impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.e[0] * t, self.e[1] * t, self.e[2] * t)
    }
}

/// Operator ovverride for Vec3*Vec3
impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new(self * v.e[0], self * v.e[1], self * v.e[2])
    }
}

/// Operator ovverride for Vec3/Vec3
impl std::ops::Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, t: f64) -> Vec3 {
        Vec3::new(self.e[0] / t, self.e[1] / t, self.e[2] / t)
    }
}

/// Ovverride of the Display trait for Vec3
impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.e[0], self.e[1], self.e[2])
    }
}
