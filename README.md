# Extrinsic Interpolator

Just a simple camera interpolator for [Gaussian Splatting](https://github.com/yindaheng98/gaussian-splatting) framework.

```sh
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade --target . --no-deps .
python -m extrinterp.render -s data/truck -d output/truck -i 30000 --load_camera output/truck/cameras.json --interp_n 300 --interp_window_size 3 --use_intrinsics dict(image_width=1600,FoVx=1.4749,image_height=1200,FoVy=1.1990)
```