use crate::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};


/// Represents a camera used for rendering scenes.
pub struct Camera {
    /// The aspect ratio of the camera (width / height).
    pub aspect_ratio: f64,
    /// The width of the image.
    pub image_width: i32,
    /// The height of the image.
    pub image_height: i32,
    /// The center point of the camera.
    pub center: Point3,
    /// The location of the pixel in the upper left corner.
    pub pixel00_loc: Vec3,
    /// The horizontal delta vector from pixel to pixel.
    pub pixel_delta_u: Vec3,
    /// The vertical delta vector from pixel to pixel.
    pub pixel_delta_v: Vec3,
    /// The count of random ray samples for each pixel (antialiasing).
    pub samples_per_pixel: i32,
    /// The maximum number of bounces from an object for a ray.
    pub max_depth: i32,
    /// The vertical view angle (field of view).
    pub vfov: f64,
}


impl Camera {
    /// Constructor of a new Camera object
    pub fn new() -> Self {
        let mut camera = Camera {
            aspect_ratio: 16.0 / 9.0,
            image_width: 1024,
            image_height: 0,
            center: Vec3::new(0.0, 0.0, 0.0),
            pixel00_loc: Vec3::new(0.0, 0.0, 0.0),
            pixel_delta_u: Vec3::new(0.0, 0.0, 0.0),
            pixel_delta_v: Vec3::new(0.0, 0.0, 0.0),
            samples_per_pixel: 10,
            max_depth: 30,
            vfov: 80.0,
        };
        camera.initialize();
        camera
    }

    /// Initializes the camera parameters.
    pub fn initialize(&mut self) {
        // Calcolare l'altezza dell'immagine e assicurarsi che sia almeno 1.
        let image_height: f64 = self.image_width as f64 / self.aspect_ratio;
        if self.image_height < 1 {
            self.image_height = image_height as i32
        } else {
            self.image_height = 1
        };
        println!("Image size: {}x{}", self.image_width, self.image_height);

        // Camera parameters
        let focal_length: f64 = 1.0;
        let theta: f64 = degrees_to_radians(self.vfov);
        let h: f64 = (theta / 2.0).tan();
        let viewport_height: f64 = 2.0 * h * focal_length;
        // let viewport_height: f64 = 2.0;
        let viewport_width: f64 =
            viewport_height * (self.image_width as f64 / self.image_height as f64);
        self.center = Vec3 { e: [0.0, 0.0, 0.0] }; // Assumendo che Vec3 sia definito

        // Calcolare i vettori lungo gli orizzonti e i bordi verticali del viewport.
        let viewport_u: Vec3 = Vec3 {
            e: [viewport_width, 0.0, 0.0],
        };
        let viewport_v: Vec3 = Vec3 {
            e: [0.0, -viewport_height, 0.0],
        };

        // Calcolare i vettori di delta orizzontali e verticali da pixel a pixel.
        self.pixel_delta_u = Vec3 {
            e: [viewport_u.e[0] / self.image_width as f64, 0.0, 0.0],
        };
        self.pixel_delta_v = Vec3 {
            e: [0.0, viewport_v.e[1] / self.image_height as f64, 0.0],
        };

        // Calcolare la posizione del pixel in alto a sinistra.
        let viewport_upper_left: Vec3 = Vec3 {
            e: [
                self.center.e[0] - 0.0,
                self.center.e[1] - 0.0,
                self.center.e[2] - focal_length,
            ],
        } - viewport_u / 2.0
            - viewport_v / 2.0;

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    /// Renders the scene and saves it to a file.
    ///
    /// # Arguments
    ///
    /// * `world` - The scene to render.
    ///
    /// # Returns
    ///
    /// An `io::Result` containing the result of the operation.
    pub fn render(&self, world: &dyn Hittable) -> Result<(), io::Error> {
        // Definisco l'intestazione del file ppm
        let mut image_string = format!("P3\n{} {}\n255\n", self.image_width, self.image_height);

        for j in 0..self.image_height {
            eprintln!("\rScanlines remaining: {}", self.image_height - j);
            for i in 0..self.image_width {
                let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
                for _ in 0..self.samples_per_pixel {
                    let r = self.get_ray(i, j);
                    pixel_color += Self::ray_color(&r, self.max_depth, world);
                }
                image_string += &write_color(pixel_color, self.samples_per_pixel);
            }
        }

        let file = File::create("output.ppm")?;
        let mut file = BufWriter::new(file);
        file.write_all(image_string.as_bytes())?;
        Ok(())
    }

    /// Calculates the color of a pixel given a ray and the scene.
    ///
    /// # Arguments
    ///
    /// * `r` - The ray to trace.
    /// * `depth` - The maximum recursion depth for ray bouncing.
    /// * `world` - The scene to trace the ray in.
    ///
    /// # Returns
    ///
    /// The color of the pixel as a `Color` struct.
    pub fn ray_color(r: &Ray, depth: i32, world: &dyn Hittable) -> Color {
        // Initialize the HitRecord
        let mut rec: HitRecord = HitRecord {
            p: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
            mat: Rc::new(Lambertian::new(Color::new(0.0, 0.0, 0.0))),
            t: 0.0,
            front_face: false,
        };

        // Set the value of infinity
        let _infinity = f64::INFINITY;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color { e: [0.0, 0.0, 0.0] };
        }

        if world.hit(
            &r,
            Interval {
                min: 0.001,
                max: f64::INFINITY,
            },
            &mut rec,
        ) {
            let mut scattered = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let mut attenuation = Color::new(0.0, 0.0, 0.0);
            if rec.mat.scatter(&r, &rec, &mut attenuation, &mut scattered) {
                return attenuation * Self::ray_color(&scattered, depth - 1, world);
            }
            return Color::new(0.0, 0.0, 0.0);
        }

        let unit_direction: Vec3 = unit_vector(r.direction());
        let a: f64 = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * Color { e: [1.0, 1.0, 1.0] } + a * Color { e: [0.5, 0.7, 1.0] };
    }

    /// Calculates a random ray that hits the pixel at position `(i, j)`.
    ///
    /// # Arguments
    ///
    /// * `i` - The horizontal position of the pixel.
    /// * `j` - The vertical position of the pixel.
    ///
    /// # Returns
    ///
    /// A `Ray` that hits the pixel at position `(i, j)`.
    pub fn get_ray(&self, i: i32, j: i32) -> Ray {
        let pixel_center: Vec3 =
            self.pixel00_loc + (i as f64 * self.pixel_delta_u) + (j as f64 * self.pixel_delta_v); // Finds the center of the pixel
        let pixel_sample: Vec3 = pixel_center + self.pixel_sample_square(); // Finds a random point inside the pixel

        let ray_origin: Vec3 = self.center; // The origin of the ray is the center of the camera
        let ray_direction: Vec3 = pixel_sample - ray_origin; // The direction of the ray is the vector from the center of the camera to the pixel_sample finded before

        Ray::new(ray_origin, ray_direction) // Returns the random finded ray
    }

    /// Calculates a random point inside a pixel.
    ///
    /// # Returns
    ///
    /// A `Vec3` representing a random point inside a pixel.
    fn pixel_sample_square(&self) -> Vec3 {
        let px = -0.5 + random_double(); // px is a random number between -0.5 and 0.5
        let py = -0.5 + random_double(); //py is a random number between -0.5 and 0.5
        (px * self.pixel_delta_u) + (py * self.pixel_delta_v) // Returns the random point
    }
}
