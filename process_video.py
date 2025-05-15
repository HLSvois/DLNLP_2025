# 视频处理逻辑，输入视频路径，返回 json 结果

from openai import OpenAI
import os
import base64
#  Base64 编码格式
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")
key='sk-4ea605511dbb43289cd2cbf0b1533559'
def process_video_file(video_path):
    # TODO: 你可以在这里进行推理、抽帧、字幕提取等操作
    print(f"处理视频：{video_path}")

    # 将xxxx/test.mp4替换为你本地视频的绝对路径
    base64_video = encode_video(video_path=video_path)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key='sk-8cc4fc618cf543dc922f07eb5fd789ef',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",  
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        # 直接传入视频文件时，请将type的值设置为video_url
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                    },
                    {"type": "text", "text":"**请你以结构化方式总结该视频的主要内容，包括以下几个方面：**\n\n1. **视频类型**（例如：教学/讲座/访谈/会议/生活记录/其他）\n2. **主要人物/说话人**（列出姓名或角色）\n3. **核心内容概要**（用简洁段落概括视频讲了什么）\n4. **关键信息要点**（用列表列出3-5个关键点）\n5. **情绪或氛围**（例如：轻松/严肃/紧张/欢快）\n6. **适合受众**（该视频适合哪些人观看）\n\n请使用简洁准确的语言生成中文输出。"},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)

    # 模拟结果
    return {
        "status": "success",
        "message": "视频处理完成",
        "video": video_path,
        "result": {
            "duration": "约4分30秒",
            "summary": completion.choices[0].message.content
        }
    }






