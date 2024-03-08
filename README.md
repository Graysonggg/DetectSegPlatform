# DetectSegPlatform
# 基于YOLO-World的目标检测及其分割管理平台

目标要素识别算法：人、车（机动车、非机动车）、物（公交站牌、地铁站出入口）识别；
基于识别要素的辅助测量测量软件交互工具：对图像中测量要素进行交互测量及标注，包括步道宽度、非机动车道宽度、机动车道宽度、过街间距、活动空间尺度等。

# 技术特性

## 深度学习

- **YOLO-World🚀**：YOLO-World是一种创新的实时开放词汇对象检测技术，它通过视觉-语言建模和大规模数据集上的预训练，提升了YOLO系列检测器的能力，使其能够在零样本情况下高效检测广泛的对象。
- **PyTorch**：机器学习框架，以动态计算图为基础，具有灵活性和易用性
- **OpenCV**：计算机视觉库，提供了丰富的图像和视频处理功能

## 前端

- **Vue3**：采用 Vue3 + script setup 最新的 Vue3 组合式 API
- **Element Plus**：Element UI 的 Vue3 版本
- **Pinia**: 类型安全、可预测的状态管理库
- **Vite**：新型前端构建工具
- **Vue Router**：路由
- **TypeScript**：JavaScript 语言的超集
- **PNPM**：更快速的，节省磁盘空间的包管理工具
- **Scss**：和 Element Plus 保持一致
- **CSS 变量**：主要控制项目的布局和颜色
- **ESlint**：代码校验
- **Prettier**：代码格式化
- **Axios**：发送网络请求
- **UnoCSS**：具有高性能且极具灵活性的即时原子化 CSS 引擎
- **注释**：各个配置项都写有尽可能详细的注释
- **兼容移动端**: 布局兼容移动端页面分辨率

## 后端

- **MySQL 8**：关系型数据库管理系统，全文索引、多源复制、更强大的JSON支持
- **Docker**：轻量级的虚拟化技术，快速构建、部署和运行应用程序
- **Flask**：用Python编写的微型Web框架
- **Werkzeug**：用于Web服务器网关接口（WSGI）应用程序的Python编程语言的实用程序库
- **SQLAlchemy**：ORM映射、SQL表达式构建、数据库连接池
- **Flask-Migrate**：数据库迁移
- **Flask-JWT-Extended**：JWT的认证和授权
- **Flask-WTF**：Web表单生成和验证功能
- **Flask-Mail**：电子邮件发送和验证
- **PyMySQL**：MySQL数据库驱动程序

<br>
