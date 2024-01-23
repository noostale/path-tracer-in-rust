#![allow(dead_code)]

extern crate rand;

use crate::rand::Rng;
use std::rc::Rc;

// Personal modules
mod vec3;
use vec3::*;

mod camera;
use camera::*;

// MAIN

/// Creates a world with some objects and renders it to a file.
fn main() {
    // World object (Instance of HittableList)
    let mut world = HittableList { objects: vec![] };

    // Create materials
    let material_ground: Rc<Lambertian> = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let material_center: Rc<Lambertian> = Rc::new(Lambertian::new(Color::new(0.7, 0.3, 0.3)));
    let material_left: Rc<Metal> = Rc::new(Metal::new(Color::new(0.8, 0.8, 0.8)));
    let material_right: Rc<Metal> = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2)));

    let material_triangle: Rc<Lambertian> = Rc::new(Lambertian::new(Color::new(0.7, 0.3, 0.3)));


    // Ricorda come funziona l'ambiente 3D:
    // Il centro del mondo è il pov della telecamera
    // Un valore positivo di x sposta a destra, negativo a sinistra
    // Un valore positivo di y sposta verso l'alto, negativo verso il basso
    // Un valore negativo di z allontana dall'osservatore

    // Create spheres
    let sphere_ground = Sphere {
        center: Vec3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        mat: material_ground,
    };
    let sphere_center = Sphere {
        center: Vec3::new(0.0, 1.0, -2.0),
        radius: 0.5,
        mat: material_center,
    };
    let sphere_left = Sphere {
        center: Vec3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        mat: material_left,
    };
    let sphere_right = Sphere {
        center: Vec3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        mat: material_right,
    };
    let triangle: Triangle = Triangle::new(
         Vec3::new(1.0, 1.0, -2.0),
         Vec3::new(1.0, 1.0, -2.0),
         Vec3::new(1.0, 1.0, -2.0),
         
         material_triangle,
    );

    // Add to the world elements
    world.add(Rc::new(sphere_ground));
    world.add(Rc::new(sphere_center));
    world.add(Rc::new(sphere_left));
    world.add(Rc::new(sphere_right));
    // world.add(Rc::new(triangle));

    let camera: Camera = Camera::new();

    // Render the image
    let _ = camera.render(&world);
}

/// Converts a linear color component in the range [0,1] to a gamma corrected value
fn linear_to_gamma(linear_component: f64) -> f64 {
    linear_component.sqrt()
}

// RANDOM NUMBER GENERATOR

/// Generates a random f64 number in the range [0,1]
fn random_double() -> f64 {
    // Returns a random real in [0,1).
    rand::thread_rng().gen_range(0.0..1.0)
}

/// Generates a random number in the range [min, max]
fn random_double_range(min: f64, max: f64) -> f64 {
    // Returns a random real in [min,max).
    rand::thread_rng().gen_range(min..max)
}

/// Outputs the color of a pixel with antialiasing
fn write_color(pixel_color: Color, samples_per_pixel: i32) -> String {
    let mut r: f64 = pixel_color.x(); // Extracts the red color components [0,255]
    let mut g: f64 = pixel_color.y(); // Extracts the green color components [0,255]
    let mut b: f64 = pixel_color.z(); // Extracts the blue color components [0,255]

    let scale = 1.0 / samples_per_pixel as f64; // Scale the color components by the number of samples
    r *= scale;
    g *= scale;
    b *= scale;

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval {
        min: 0.0,
        max: 0.999,
    }; // Clamp the intensity of the color components to [0,1]

    let r: i32 = (256.0 * intensity.clamp(r)) as i32;
    let g: i32 = (256.0 * intensity.clamp(g)) as i32;
    let b: i32 = (256.0 * intensity.clamp(b)) as i32;

    format!("{} {} {}\n", r, g, b)
}


// RAY STRUCTURE

/// Rappresents a Ray in the 3D space
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// Origin point of the ray
    orig: Point3,
    /// Direction vector of the ray
    dir: Vec3,
}

/// Implementation of the methods for the Ray structure
impl Ray {
    /// Constructor for the Ray structure
    pub fn new(orig: Point3, dir: Vec3) -> Self {
        return Self {
            orig: orig,
            dir: dir,
        };
    }

    /// Getter for the Ray origin
    pub fn origin(&self) -> Point3 {
        return self.orig;
    }

    /// Getter for the Ray direction
    pub fn direction(&self) -> Vec3 {
        return self.dir;
    }

    pub fn at(&self, t: f64) -> Point3 {
        return self.orig + t * self.dir;
    }
}

// HITRECORD STRUCTURE

/// Contains the informations about the intersection between a Ray and a Point
#[derive(Clone)]
pub struct HitRecord {
    /// Intersection Point between a Point and a Ray
    p: Vec3,

    /// Normal vector between a Point and a Ray
    normal: Vec3,

    /// Material of the object hitted by a Ray
    mat: Rc<dyn Material>,

    /// Parameter t that intersects the Point and the object using line equation
    t: f64,

    /// Boolean to set if the the intersection point is onward or outward
    front_face: bool,
}

/// Implementation of the methods of the structure HitRecord
impl HitRecord {
    /// Constructor of a new HitRecord object
    pub fn new(p: Vec3, normal: Vec3, mat: Rc<dyn Material>, t: f64, front_face: bool) -> Self {
        HitRecord {
            p,
            normal,
            mat,
            t,
            front_face,
        }
    }

    /// Method to set the normal of the HitRecord structure
    fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        // Imposta il vettore normale del record di collisione.
        // NOTA: si assume che il parametro `outward_normal` abbia lunghezza unitaria.

        self.front_face = r.direction().dot(&outward_normal) < 0.0; // Assumendo che sia definita la funzione dot per Vec3
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

// MATERIAL TRAIT

/// Trait for the materials, we need to define how a ray can be scattered by a material
pub trait Material {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool;
}

// HITTABLE TRAIT

/// Trait for the Hittable objects, we need to define how on object can be hitted by a ray
pub trait Hittable {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool;
}

// SPHERE STRUCTURE

/// Rappresents a sphere
struct Sphere {
    /// Center of the sphere
    center: Point3,

    /// Radius of the sphere
    radius: f64,

    /// Material of the sphere
    mat: Rc<dyn Material>,
}

/// Implementation of the methods for Sphere
impl Sphere {
    /// Constructor of a new Sphere object
    fn new(center: Point3, radius: f64, mat: Rc<dyn Material>) -> Sphere {
        Sphere {
            center,
            radius,
            mat,
        }
    }
}

/// Implementations of the trait Hittable to check how a Shere can be hitted by a Ray, returns true or false
impl Hittable for Sphere {
    /// Method to check if a Ray hit a Sphere
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let oc: Point3 = r.origin() - self.center;
        let a: f64 = r.direction().length_squared();
        let half_b: f64 = oc.dot(&r.direction());
        let c: f64 = oc.length_squared() - self.radius * self.radius;

        let discriminant: f64 = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return false;
        }
        let sqrtd = discriminant.sqrt();

        // Finds the nearest root that lies in the acceptable range.
        let mut root: f64 = (-half_b - sqrtd) / a;
        if root <= ray_t.min || ray_t.max <= root {
            root = (-half_b + sqrtd) / a;
            if root <= ray_t.min || ray_t.max <= root {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(&r, outward_normal);
        rec.mat = self.mat.clone();

        true
    }
}

/// Rappresents a triangle
struct Triangle {
    /// Vertex 1 of the triangle
    vertex0: Point3,

    /// Vertex 2 of the triangle
    vertex1: Point3,

    /// Vertex 3 of the triangle
    vertex2: Point3,

    /// Material of the triangle
    mat: Rc<dyn Material>,
}

/// Implementation of the methods for Triangle
impl Triangle {
    /// Constructor of a new Triangle object
    fn new(vertex0: Point3, vertex1: Point3, vertex2: Point3, mat: Rc<dyn Material>) -> Triangle {
        Triangle {
            vertex0,
            vertex1,
            vertex2,
            mat,
        }
    }
}

/// Implementations of the trait Hittable to check how a Triangle can be hitted by a Ray, returns true or false
impl Hittable for Triangle {
    /**
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        // Vectors of the edges of the triangle
        let edge1: Vec3 = self.vertex2 - self.vertex1; // x,y,z distance between vertex1 and vertex2
        let edge2: Vec3 = self.vertex3 - self.vertex1; // x,y,z distance between vertex1 and vertex3
        let normal: Vec3 = unit_vector(cross(edge1, edge2)); // Normal vector of the triangle

        let denom: f64 = dot(normal, r.direction()); // scalar product between the normal and the direction of the ray to check the allignment of the ray with the plane of the triangle

        if denom.abs() < 1e-6 {
            // if the ray is parallel to the plane of the triangle (near zero means parallel)
            return false;
        }

        let t: f64 = dot(self.vertex1 - r.origin(), normal) / denom; // finds the t parameter of the ray equation (ray = origin + t*direction)

        // Check if the intersection is in the ray interval (t too big or too small)
        if !ray_t.contains(t) {
            return false;
        }

        // Controlla se il punto di intersezione è all'interno del triangolo
        let edge1_cross_p = cross(edge1, r.orig - self.vertex1);

        if dot(normal, edge1_cross_p) < 0.0 {
            return false;
        }

        let edge2_cross_p = cross(edge2, r.orig - self.vertex1);
        if dot(normal, edge2_cross_p) < 0.0 {
            return false;
        }

        // If the intersection is in the ray interval and inside the triangle, then it is a hit, populate the HitRecord structure
        rec.t = t;
        rec.p = r.at(rec.t);
        rec.mat = self.mat.clone();
        rec.set_face_normal(r, normal);

        true
    }
    */
    
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        // Compute the plane's normal
        let v0v1 = self.vertex1 - self.vertex0;
        let v0v2 = self.vertex2 - self.vertex0;
        let normal = unit_vector(cross(v0v1, v0v2)); // Normal vector of the triangle

        // Check if the ray and plane are parallel
        let ndot_ray_direction = normal.dot(&r.direction());
        if ndot_ray_direction.abs() < 1e-6 {
            return false; // They are parallel, so they don't intersect
        }

        // Compute d parameter using equation 2
        let d = -normal.dot(&self.vertex0);

        // Compute t (equation 3)
        let t = -(normal.dot(&r.origin()) + d) / ndot_ray_direction;

        // Check if the triangle is behind the ray
        if t < ray_t.min || t > ray_t.max {
            return false; // The triangle is behind
        }

        // Compute the intersection point using equation 1
        let intersection_point = r.at(t);

        // Inside-outside test
        let mut c: Vec3;

        // Edge 0
        let edge0 = self.vertex1 - self.vertex0;
        let vp0 = intersection_point - self.vertex0;
        c = cross(edge0, vp0);
        if normal.dot(&c) < 0.0 {
            return false; // P is on the right side
        }

        // Edge 1
        let edge1 = self.vertex2 - self.vertex1;
        let vp1 = intersection_point - self.vertex1;
        c = cross(edge1, vp1);
        if normal.dot(&c) < 0.0 {
            return false; // P is on the right side
        }

        // Edge 2
        let edge2 = self.vertex0 - self.vertex2;
        let vp2 = intersection_point - self.vertex2;
        c = cross(edge2, vp2);
        if normal.dot(&c) < 0.0 {
            return false; // P is on the right side
        }

        // If the intersection is within the ray interval and inside the triangle, populate the HitRecord structure
        rec.t = t;
        rec.p = intersection_point;
        rec.mat = self.mat.clone();
        rec.set_face_normal(r, normal);

        true
    }
}
    



// HITTABLELIST STRUCTURE

/// Rappresents a list of Hittable objects (world)
///
/// The objects of the world are stored in a HittableList object (Vec of Hittable objects)

struct HittableList {
    objects: Vec<Rc<dyn Hittable>>, // Vector of Hittable objects
}

/// Implementation of methods for HittableList
impl HittableList {
    /// Constructor of a new HittableList object
    fn new() -> HittableList {
        HittableList { objects: vec![] }
    }

    /// Method to add any kind of object that implements Hittable
    fn add(&mut self, object: Rc<dyn Hittable>) {
        self.objects.push(object);
    }

    /// Method to clear the list
    fn clear(&mut self) {
        self.objects.clear();
    }
}

/// Implementation of the Hittable trait for the HittableList
impl Hittable for HittableList {
    /// Method to check if a Ray hit any object of the world
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec: HitRecord = HitRecord::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Rc::new(Lambertian::new(Color::new(0.0, 0.0, 0.0))),
            0.0,
            false,
        );
        let mut hit_anything: bool = false;
        let mut closest_so_far: f64 = ray_t.max;

        for object in &self.objects {
            // for every oject in the world
            if object.hit(
                r,
                Interval {
                    min: ray_t.min,
                    max: closest_so_far,
                },
                &mut temp_rec,
            ) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec.clone();
            }
        }

        hit_anything
    }
}

/// Fuction to convert f64 degrees to f64 radiants
fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

// INTERVAL STRUCTURE

/// Rappresents an interval between two values min and max
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    min: f64, // Minimum value of the interval
    max: f64, // Maximum value of the interval
}

/// Implementation of methods for Interval
impl Interval {
    pub const EMPTY: Interval = Interval {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    pub const UNIVERSE: Interval = Interval {
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
    };

    /// Creates an empty interval.
    pub fn new_empty() -> Self {
        Self::EMPTY
    }

    /// Creates an interval with specific min and max.
    pub fn new(min: f64, max: f64) -> Self {
        Interval { min, max }
    }

    /// Checks if the interval contains x (inclusive).
    pub fn contains(&self, x: f64) -> bool {
        self.min <= x && x <= self.max
    }

    /// Checks if the interval surrounds x (exclusive).
    pub fn surrounds(&self, x: f64) -> bool {
        self.min < x && x < self.max
    }

    /// Limits the input to the min and max of the interval.
    pub fn clamp(&self, x: f64) -> f64 {
        if x < self.min {
            return self.min;
        }
        if x > self.max {
            return self.max;
        }
        x
    }
}

/// Rappresents a Lambertian material
#[derive(Debug, Copy, Clone)]
pub struct Lambertian {
    /// Albedo of the material (refleve power of the material)
    albedo: Color,
}

/// Implementation of the Lambertian material
impl Lambertian {
    /// Constructor of a new Lambertian material
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

/// Implementazion of Material trait for Lambertian material type (Opaque)
impl Material for Lambertian {
    /// Method to calculate the scattering of a ray on a Lambertian material
    fn scatter(
        &self,
        _r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = rec.normal + Vec3::random_unit_vector();

        // Controllo per direzione di scattering degenerata
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        *scattered = Ray::new(rec.p, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}


/// Rappresents a Metal material
pub struct Metal {
    /// albedo value of the Metal 
    albedo: Color,
}

/// Implementation of the Metal structure
impl Metal {

    /// Constructor of a new Metal material
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

/// Function to calculate the reflection of a vector on a surface
fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(&n) * n //
}


/// Implementations for the Metal structure
impl Material for Metal {

    /// Method to calculate the scattering of a ray on a Metal material
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray) -> bool {


        let reflected = reflect(unit_vector(r_in.dir), rec.normal);
        *scattered = Ray {
            orig: rec.p,
            dir: reflected,
        };
        *attenuation = self.albedo;
        true
    }
}
