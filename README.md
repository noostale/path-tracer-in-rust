# How to compile and run

## Compiling

To compile the program, run the following command in the root directory of the project:

```bash
cargo build
```


If you want to compile the program without debbuging information, run the following command in the root directory of the project:

```bash
cargo build --release
```

Both methods will create a binary in the `target/release` directory. The binary will be named `path_tracer`.

## Building documentation

To build the documentation, run the following command in the root directory of the project:

```bash
cargo doc
```

The documentation will be created in the `target/doc` directory. To view the documentation, open the `index.html` file in the `target/doc` directory in a web browser or run the following command in the root directory of the project:

```bash
cargo doc --open
```

## Running

To run the program, run the following command in the root directory of the project:

```bash
cargo run
```

To run the program without debbuging information, run the following command in the root directory of the project:

```bash
cargo run --release
```

The program will create a `output.ppm` file in the root directory of the project. The file will contain the rendered image.

# Output examples

[output.ppm](output.ppm) is an example of the output of the program.
