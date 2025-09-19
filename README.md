# Data Compression Using SVD

## Project Description

This project demonstrates image compression using **Singular Value Decomposition (SVD)**. By leveraging the mathematical properties of SVD, we can approximate images with fewer singular values, significantly reducing storage requirements while maintaining visual quality. The project allows users to experiment with different compression levels and visualize the trade-off between image quality and compression ratio.

## Features

- Compress grayscale images using low-rank SVD approximations
- Experiment with different numbers of singular values (ranks)
- Visualize original vs. compressed images side-by-side
- Plot quality vs. compression ratio
- Simple, modular code for easy experimentation

## Tech Stack

- **Python**
- **NumPy**
- **Matplotlib**
- **Pillow** or **OpenCV** (for image I/O)

## Installation & Usage

1. **Clone the repository:**
	 ```bash
	 git clone https://github.com/SaaiAravindhRaja/CS103-G1T5.git
	 cd CS103-G1T5
	 ```

2. **Install dependencies:**
	 ```bash
	 pip install numpy matplotlib pillow opencv-python
	 ```

3. **Run the notebook or script:**
	 - Open `svd_compression.ipynb` in Jupyter Notebook **OR**
	 - Run the script:
		 ```bash
		 python svd_compression.py
		 ```

## Examples / Results

Below are sample results of compressing an image using different numbers of singular values:

| Original Image | 10 Singular Values | 50 Singular Values | 100 Singular Values |
|:--------------:|:-----------------:|:------------------:|:-------------------:|
| ![](results/original.png) | ![](results/svd_10.png) | ![](results/svd_50.png) | ![](results/svd_100.png) |

## Future Improvements

- Add quantitative metrics (e.g., PSNR, SSIM) for quality assessment
- Support for color image compression
- Interactive web UI for real-time experimentation
- Batch processing of multiple images

## Contributors

<table>
	<tr>
			<td align="center">
				<a href="https://github.com/SaaiAravindhRaja">
					<img src="https://github.com/SaaiAravindhRaja.png" width="80" alt="SaaiAravindhRaja"/><br/>
					<sub><b>SaaiAravindhRaja</b></sub><br/>
					<sub>Saai</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/halowenfright">
					<img src="https://github.com/halowenfright.png" width="80" alt="halowenfright"/><br/>
					<sub><b>halowenfright</b></sub><br/>
					<sub>Sherman</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/ravenglider">
					<img src="https://github.com/ravenglider.png" width="80" alt="ravenglider"/><br/>
					<sub><b>ravenglider</b></sub><br/>
					<sub>Sonia</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/cohiee">
					<img src="https://github.com/cohiee.png" width="80" alt="cohiee"/><br/>
					<sub><b>cohiee</b></sub><br/>
					<sub>Vincent</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/seraphiii">
					<img src="https://github.com/seraphiii.png" width="80" alt="seraphiii"/><br/>
					<sub><b>seraphiii</b></sub><br/>
					<sub>Zaccheus</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/Ridheema776">
					<img src="https://github.com/Ridheema776.png" width="80" alt="Ridheema776"/><br/>
					<sub><b>Ridheema776</b></sub><br/>
					<sub>RIdheema</sub>
				</a>
			</td>
	</tr>
</table>
# CS103-G1T5