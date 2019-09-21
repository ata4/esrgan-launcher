This script is a ESRGAN implementation based on the [original ESRGAN repository](https://github.com/xinntao/ESRGAN) for upscaling images. It's primarily meant to replace the very rudimentary test.py of the original repo with a more powerful and intuitive command line interface and close the gap between the two ESRGAN model architectures.

## Models

The script differentiates between two types of models: primary models (``--model``) and filter models. Filter models also exist in two variants: pre-filter (``--pre-filter``) and post-filter (``--post-filter``). Filter models are typically 1x scale models, but they don't necessarily have to.

The file processing works like this: for each primary model, go through each input file, then apply all pre-filter models, then the current primary model, then all post-filter models and write the result to the output directory. Each primary model will have its own output sub-directory, which is named by the chain of models being used.

Both the "old arch" and the "new arch" ESRGAN model format is supported. The scale is determined from the model dict structure and therefore doesn't have to and, in fact, can't be specified manually. Scales other than 4x are supported for the "new arch" format, although only in theory, since I haven't encountered any model files for scales other than 4x in the wild.

Models can be interpolated on the fly by providing a list of path/weight pairs, separated by path separators. For example, for a 25/25/50 mix of three models, the argument must be specified as followed:

```
--model model1.pth:25:model2.pth:25:model3.pth:50
```

The path separator depends on the platform. It's ":" on Unix/Linux/MacOS and ";" on Windows.

Note that the weights don't have to add up to 100, since the model alpha is calculated by the weight divided by the total weight. So in the example above, it's perfectly fine to use 1/1/2 instead. Also keep in mind that not all models can be interpolated without issues.

For a list of freely available models to download, take a look at the [upscale.wiki model database](https://upscale.wiki/wiki/Model_Database).

## Tiling

Since upscaling images can take a lot of (V)RAM, the script splits large input images into smaller tiles, upscales those one by one and puts them back together in memory by default. Some overlapping is applied to compensate artifacts near the border of the tiles. The result still varies slightly compared to upscaling the image in one go, but it should be barely noticeable. Still, it's recommended to experiment with the ``--tilesize`` argument to find the maximum value that your GPU supports to reduce its impact on the output.

## Command line examples

Upscale all images in folder `input` and put the result in `output` using all files from folder `models`.

```
main.py input output --model models
```

---

Upscale image `teapot.png` in folder `input` and put the result in `output` using all .pth files from folder `models` starting with `4x`. Keep in mind that quotes are required for the path for wildcards to work!

```
main.py input/teapot.png output --model "models/4x*.pth"
```

---

Upscale all images in folder `input` and put the result in `output` using the model files `Manga109Attempt.pth` and `RRDB_ESRGAN_x4.pth`.

```
main.py input output --model models/Manga109Attempt.pth --model models/RRDB_ESRGAN_x4.pth
```

---

Upscale all images in folder `input` and put the result in `output` using a 25/75 interpolation between `Manga109Attempt.pth` and `RRDB_ESRGAN_x4.pth` as model under Linux (Windows: use `;` instead of `:`).

```
main.py input output --model models/Manga109Attempt.pth:25:models/RRDB_ESRGAN_x4.pth:75
```

---

Upscale all images in folder `input` and put the result in `output`. Apply the filter models `1x_JPEG_40-60.pth` and `1x_normals_generator_general_215k.pth`, then upscale using the model file `RRDB_ESRGAN_x4.pth` and finally, apply the filter model `1x_DeSharpen.pth`.

```
main.py input output --prefilter models/1x_JPEG_40-60.pth --prefilter models/1x_normals_generator_general_215k.pth --model models/RRDB_ESRGAN_x4.pth --postfilter models/1x_DeSharpen.pth
```

---

Upscale all images in folder `input` and put the result in `output` using all files from folder `models` with GPU acceleration and a tile size of 1024 pixels.

```
main.py input output --device cuda --tilesize 1024 --model models
```

## Docker

A Dockerfile and docker-compose file is included to easily create a pyTorch runtime environment for the script.

The recommended command to run the container is:

```
docker-compose run --rm main [arguments]
```

For CPU-only mode, use this command instead:

```
docker-compose run --rm main-cpu [arguments]
```

Note that the scripts and models are not integrated into the container by default and are mounted from the base directory as a volume for easy editing instead.
