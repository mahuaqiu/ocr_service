@echo off
REM OCR Service Docker Compose 启动脚本

REM 启动服务
docker-compose up -d

echo OCR Service 启动完成
echo 访问地址: http://localhost:9021
echo API 文档: http://localhost:9021/docs
echo.
echo 查看日志: docker-compose logs -f
echo 停止服务: docker-compose down