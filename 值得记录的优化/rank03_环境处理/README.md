# 情景:部署环境中无timm
睿抗的测试机没有timm,如果从外部导入,正确性很难保证不会跌.所以考虑使用transformer重写.  
## 细节
- timm→transformers 权重转换(qkv 拆分、key 映射、layer_scale→lambda1)
- 无损验证:转换后 F1 与 timm 版逐项一致
- cv2 BGR→RGB→resize224+ImageNet normalize 适配官方 predict 接口
## 重点
不是具体 key 映射(那是 DINOv2 专属),而是**"训练用 timm、部署环境无 timm 时,用 transformers 重写 + 权重转换 + 无损验证"这个工程范式**,下次换任何模型遇到部署环境依赖缺失,这个套路通用.