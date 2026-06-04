# file_operate.py
def read_txt_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"读取失败：{str(e)}"

if __name__ == "__main__":
    # 同级目录新建 test.txt 写入任意文本
    text = read_txt_file("/Users/xufeifan/ai-dev-12w/test.txt")
    print("文件内容：", text)   