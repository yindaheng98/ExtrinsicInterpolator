# Extrinsic Interpolator

Just a simple camera interpolator for [Gaussian Splatting](https://github.com/yindaheng98/gaussian-splatting) framework.

## Install

Install from PyPI:

```sh
pip install extrinterp
pip install "extrinterp[visualizer]"
```

Install from source for development:

```sh
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --upgrade --target . --no-deps .
pip install open3d
```

`open3d` is only needed for the visualizer tools. It is also available through the optional dependency:

```sh
pip install ".[visualizer]"
```

## Render Interpolation

```sh
python -m extrinterp.renderer.interpolator -s data/truck -d output/truck -i 30000 --load_camera output/truck/cameras.json --interp_n 300 --interp_window_size 3 --use_intrinsics dict(image_width=1600,FoVx=1.4749,image_height=1200,FoVy=1.1990)
```

## Visualize Interpolation

```sh
python -m extrinterp.visualizer.interpolator -s data/truck --load_camera output/truck/cameras.json --interp_n 300 --interp_window_size 3
```

This shows the sorted input camera poses and the interpolation output in Open3D.

## Visualize Prediction

```sh
python -m extrinterp.visualizer.predictor -s data/truck --load_camera output/truck/cameras.json --update_interval 10 --predict_n 6 --integration_step 0.01 --smoothing_window 30
```

This uses `ConstantAngularAccelerationExtrinsicPredictor` on the sorted input camera poses. The black line is the ground truth path, and each orange branch is a prediction from one update point.