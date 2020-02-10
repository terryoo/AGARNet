# AGARNet: Adaptively Gated JPEG Compression Artifacts Removal Network for a Wide Range Quality Factor (IEEE Accesss 2020)

## Paper
[**IEEE Access**](https://ieeexplore.ieee.org/document/8967086)

<img src = "/figs/st_jpeg.png" width="280"> <img src = "/figs/st_sr_jpeg.png" width="280"> <img src = "/figs/st_sr_proposed.png" width="280">
<img src = "/figs/ronda_jpeg.png" width="280"> <img src = "/figs/ronda_jpeg_hdr.png" width="280"> <img src = "/figs/ronda_jpeg_hdr_processed.png" width="280">


## Test Code
[**Code**]()

[**Trained Model**](https://drive.google.com/drive/folders/1wMxa2Y4vrHDHjBOpyCp23zHHWeShyY8R?usp=sharing)

## Abstract
Most of existing compression artifacts reduction methods focused on the application for low-quality images and usually assumed a known compression quality factor. However, images compressed with high quality should also be manipulated because even small artifacts become noticeable when we enhance the compressed image. Also, the use of quality factor from the decoder is not practical because there are too many recompressed or transcoded images whose quality factor are not reliable and spatially varying. To address these issues, we propose a quality-adaptive artifacts removal network based on the gating scheme, with a quality estimator that works for a wide range of quality factor. 
Specifically, the estimator gives a pixel-wise quality factor, and our gating scheme generates gate-weights from the quality factor. Then, the gate-weights control the magnitudes of feature maps in our artifacts removal network. 
Thus, our gating scheme guarantees the proposed network to perform adaptively without changing the parameters according to the change of quality factor. Moreover, we exploit the Discrete Cosine Transform (DCT) scheme with 3D convolution for capturing both spatial and frequency dependencies of images. Experiments show that the proposed network provides better performance than the state-of-the-art methods over a wide range of quality factor. Also, the proposed method provides robust results in real-world scenarios such as the manipulation of transcoded images and videos.

## Network Architecture

<img src = "/figs/Flow.png" width="900">
The architecture of AGARNet.
<img src = "/figs/resblock3.png" width="300">
The proposed 3DG-ResBlock.

## Experimental Results
### Single JPEG Compression
<img src = "/figs/jpeg_gain.PNG" width="450">
<img src = "/figs/jpeg_results.PNG" width="900">


### JPEG Recompression
<img src = "/figs/recompression.PNG" width="450">
<img src = "/figs/recompression_figure.png" width="900">


### Video to JPEG Recompression
<img src = "/figs/transcoded.PNG" width="450">


