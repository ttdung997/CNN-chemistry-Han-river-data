# Tổng quan mã nguồn dự án lọc ảnh
## _The Last Markdown Editor, Ever_

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)


## Công nghệ sử dụng 
- Python3
- numpy, pandas
- Sklearn, Tensorflow 

## Danh sách thư mục quan trọng 
  - data: chứa dữ liệu, bao gồm hình EEM và nồng độ các chất
  - Heatmap: Chứa mã nguồn các ảnh nhiệt được sinh ra
  
## Danh sách tệp quan trọng:
  - main.py: Đọc và trích rút ảnh nhiệt cơ bản
  - linear.py: Thực hiện phân tích bằng trích chọn đặc trưng + hồi quy tuyến tính cơ bản
  - attention.py: Thực hiện dùng CNN-attention để phân tích
  - attention2.py: Tách attention thành một lớp tiêng biệt, đồng thới nhúng attention vào ảnh nhiệt
  - attention-corr.py: Trích chọn vector attention, sau đó cho vào để tính toán độ tương quan với từng chất