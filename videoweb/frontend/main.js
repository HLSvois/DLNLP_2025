document.getElementById('uploadBtn').onclick = async () => {
    const input = document.getElementById('videoInput')
    const file = input.files[0]
    if (!file) {
      alert("请先选择视频文件")
      returnw
    }
  
    const formData = new FormData()
    formData.append('video', file)
  
    const response = await fetch('http://127.0.0.1:5000/upload', {
      method: 'POST',
      body: formData
    })

  
    const result = await response.json()
    if (response.ok) {
      alert('上传成功！');
    } else {
        alert('上传失败: ' + result.error);
    }
    
    document.getElementById('resultBox').textContent = JSON.stringify(result, null, 2)
  }
  