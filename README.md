<div align="center">
<a href="https://github.com/AbdallahHemdan/Orchestra" rel="noopener">
  
  ![Component 16](https://user-images.githubusercontent.com/40190772/104846822-22d3e800-58e5-11eb-9c6c-b7de610bd483.png)


</div>

<h3 align="center">Orchestra</h3>

<div align="center">
  
  [![GitHub contributors](https://img.shields.io/github/contributors/AbdallahHemdan/Orchestra)](https://github.com/AbdallahHemdan/Orchestra/contributors)
  [![GitHub issues](https://img.shields.io/github/issues/AbdallahHemdan/Orchestra)](https://github.com/AbdallahHemdan/Orchestra/issues)
  [![GitHub forks](https://img.shields.io/github/forks/AbdallahHemdan/Orchestra)](https://github.com/AbdallahHemdan/Orchestra/network)
  [![GitHub stars](https://img.shields.io/github/stars/AbdallahHemdan/Orchestra)](https://github.com/AbdallahHemdan/Orchestra/stargazers)
  [![GitHub license](https://img.shields.io/github/license/AbdallahHemdan/Orchestra)](https://github.com/AbdallahHemdan/Orchestra/blob/master/LICENSE)
  <img src="https://img.shields.io/github/languages/count/AbdallahHemdan/Orchestra" />
  <img src="https://img.shields.io/github/languages/top/AbdallahHemdan/Orchestra" />
  <img src="https://img.shields.io/github/languages/code-size/AbdallahHemdan/Orchestra" />
  <img src="https://img.shields.io/github/issues-pr-raw/AbdallahHemdan/Orchestra" />

</div>

## About
> **Orchestra** is a sheet music reader (optical music recognition (**OMR**) system) that converts sheet music to a machine-readable version.

<div align="center">

![image](https://user-images.githubusercontent.com/40190772/104846946-e81e7f80-58e5-11eb-8652-e54b86b46fe1.png)

</div>

## How it works
> List of steps we take to process the input sheet and get our results


### 1. Noise Removal

<div align="center">

![1  noise_removed](https://user-images.githubusercontent.com/40190772/104847172-397b3e80-58e7-11eb-821f-33a83ee60416.png)

</div>


### 2. Binarization

<div align="center">

![2  binarized](https://user-images.githubusercontent.com/40190772/104847174-3aac6b80-58e7-11eb-8c85-eb9747a7c786.png)

</div>


### 3. Staff line removal

<div align="center">

![3  cleaned](https://user-images.githubusercontent.com/40190772/104847175-3b450200-58e7-11eb-8f47-1485b142e434.png)

</div>

### 4. Cutted buckets

<div align="center">

<hr />

![4  cutted-1](https://user-images.githubusercontent.com/40190772/104847181-3f711f80-58e7-11eb-83b4-435373642c8d.png)

<br /><hr />
![4  cutted-2](https://user-images.githubusercontent.com/40190772/104847179-3ed88900-58e7-11eb-8fbe-25a484c63092.png)

<br /><hr />

![4  cutted-3](https://user-images.githubusercontent.com/40190772/104847180-3ed88900-58e7-11eb-959f-817388bade77.png)

<br /><hr />
</div>

### 5. Segmentation and detection

<div align="center">
  
![colored_0_1](https://user-images.githubusercontent.com/40190772/104849087-97f8ea80-58f0-11eb-9b4d-49172eb9d9a5.png)

<br />

![colored_0_2](https://user-images.githubusercontent.com/40190772/104849089-992a1780-58f0-11eb-9fb6-0c0cc6e6dac0.png)

<br />

![colored_0_3](https://user-images.githubusercontent.com/40190772/104849090-99c2ae00-58f0-11eb-9876-4eea7f322e83.png)

  
</div>

### 6. Recognition

1. Cutted 1
> [ \meter<"4/4"> d1/4 e1/32 e2/2 e1/8 e1/16 e1/32 {e1/4,g1/4} e1/4 e1/8 c1/8 g1/32 c1/16 e1/32 ]

2. Cutted 2
> [ \meter<"4/4"> {e1/4,g1/4,b1/4} a1/8 d1/8 c1/16 g1/16 d1/16 e1/16 c2/16 g2/16 d2/16 e2/16 {f1/4,g1/4,b1/4} c1/4 a1/4. a1/8 a1/32.. ]

3. Cutted 3
> [ \meter<"4/4"> e1/16 e1/16 e1/16 e1/16 e1/4 e#1/4 g1/4 g&&1/4 g1/4 e#2/4 ]


### Installation

1. **_Clone the repository_**

```sh
$ git clone https://github.com/AbdallahHemdan/Orchestra.git
```
2. **_Navigate to repository directory_**
```sh
$ cd Orchestra
```
3. **_Install dependencies_**
```sh
$ pip install -r requirements.txt
```

### Running

1. **_Put you input files inside input folder_**
2. **_Put you output files inside output folder_**

3. **_Running_**
```sh
python main.py $path_of_input_folder $path_of_output_folder
```

## Contributing

> Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Check out our [contributing guidelines](https://github.com/AbdallahHemdan/Orchestra/blob/master/CONTRIBUTING.md) for ways to contribute.

### Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/AbdallahHemdan"><img src="https://avatars1.githubusercontent.com/u/40190772?s=460&v=4" width="150px;" alt=""/><br /><sub><b>Abdallah Hemdan</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/AdelRizq"><img src="https://avatars2.githubusercontent.com/u/40351413?s=460&v=4" width="150px;" alt=""/><br /><sub><b>Adel Mohamed</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/kareem3m"><img src="https://avatars0.githubusercontent.com/u/45700579?s=400&v=4" width="150px;" alt=""/><br /><sub><b>Kareem Mohamed^3</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/Mahboub99"><img src="https://avatars3.githubusercontent.com/u/43186742?s=460&v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Mahboub</b></sub></a><br /></td>
  </tr>
 </table>

### Licence
[MIT Licence](https://github.com/AbdallahHemdan/Orchestra/blob/master/LICENSE)
