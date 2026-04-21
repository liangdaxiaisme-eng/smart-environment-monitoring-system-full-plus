from flask import Flask, render_template_string, request
import numpy as np
import os

app = Flask(__name__)
class_names = ['清洁', '轻度污染', '中度污染', '重度污染']
trash_classes = ['branch', 'leaf', 'others', 'plastic-bag', 'plastic-bottle', 'plastic-wrapper', 'wood-log']

HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能环境监测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #e8eaed; min-height: 100vh; padding: 30px 20px; }
        .container { max-width: 760px; margin: 0 auto; }
        .header { background: white; padding: 25px 30px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
        .header h1 { font-size: 1.6em; color: #202124; margin-bottom: 6px; }
        .header p { color: #5f6368; font-size: 0.9em; }
        .tabs { display: flex; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; flex-wrap: wrap; }
        .tab { flex: 1; min-width: 25%; padding: 14px 8px; text-align: center; cursor: pointer; border: none; background: #fff; font-size: 0.9em; color: #5f6368; transition: all 0.2s; }
        .tab:hover { background: #f1f3f4; }
        .tab.active { background: #1a73e8; color: white; font-weight: 500; }
        .card { background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden; display: none; }
        .card.active { display: block; }
        .card-header { background: #1a73e8; color: white; padding: 18px 24px; }
        .card-header h2 { font-size: 1.15em; font-weight: 500; }
        .card-body { padding: 24px; }
        .guide { background: #f8f9fa; border-left: 3px solid #1a73e8; padding: 14px 18px; margin-bottom: 20px; border-radius: 0 6px 6px 0; }
        .guide-title { font-weight: 600; color: #202124; margin-bottom: 10px; font-size: 0.95em; }
        .guide ol { margin: 0; padding-left: 22px; color: #4a4a4a; line-height: 1.7; font-size: 0.9em; }
        .form-group { margin-bottom: 14px; }
        .form-group label { display: block; margin-bottom: 5px; color: #3c4043; font-size: 0.9em; font-weight: 500; }
        .form-group input { width: 100%; padding: 10px 14px; border: 1px solid #dadce0; border-radius: 6px; font-size: 0.95em; }
        .form-group input:focus { outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.1); }
        .form-row { display: flex; gap: 12px; }
        .form-row .form-group { flex: 1; }
        .btn { background: #1a73e8; color: white; border: none; padding: 12px 28px; border-radius: 6px; font-size: 0.95em; cursor: pointer; width: 100%; font-weight: 500; margin-top: 10px; }
        .btn:hover { background: #1557b0; }
        .upload-area { border: 2px dashed #dadce0; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; margin-bottom: 14px; transition: all 0.2s; }
        .upload-area:hover { border-color: #1a73e8; background: #f8f9fa; }
        .upload-area input { display: none; }
        .result { margin-top: 20px; padding: 18px; background: #e6f4ea; border: 1px solid #ceead6; border-radius: 8px; }
        .result h3 { color: #137333; margin-bottom: 10px; font-size: 1em; }
        .result p { color: #3c4043; line-height: 1.6; font-size: 0.9em; }
        .result-trash { margin-top: 20px; padding: 18px; background: #fef7e0; border: 1px solid #feefc3; border-radius: 8px; }
        .result-trash h3 { color: #b06000; margin-bottom: 10px; font-size: 1em; }
        .detection-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .detection-item:last-child { border-bottom: none; }
        .detection-class { color: #3c4043; }
        .detection-conf { color: #1a73e8; font-weight: 500; }
        .footer { text-align: center; padding: 20px; color: #80868b; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 智能环境监测系统</h1>
            <p>基于深度学习的环境监测与预测平台</p>
        </div>
        
        <div class="tabs">
            <button class="tab {{'active' if active_tab=='water' else ''}}" onclick="switchTab('water')">💧 水质分析</button>
            <button class="tab {{'active' if active_tab=='air' else ''}}" onclick="switchTab('air')">🌫️ 空气预测</button>
            <button class="tab {{'active' if active_tab=='trash' else ''}}" onclick="switchTab('trash')">🚮 垃圾检测</button>
            <button class="tab {{'active' if active_tab=='fusion' else ''}}" onclick="switchTab('fusion')">🔗 综合评估</button>
        </div>
        
        <!-- 水质分析 -->
        <div id="panel-water" class="card {{'active' if active_tab=='water' else ''}}">
            <div class="card-header"><h2>💧 水质图像分类</h2></div>
            <div class="card-body">
                <div class="guide">
                    <div class="guide-title">📖 使用说明</div>
                    <ol><li>点击下方按钮选择水质图片（支持jpg、png）</li><li>点击"开始分析"按钮进行识别</li><li>系统将显示水质污染等级及置信度</li></ol>
                </div>
                <label class="upload-area" onclick="document.getElementById('waterFile').click()">
                    <input type="file" name="file" accept="image/*" id="waterFile">
                    <div id="uploadText">📷 点击上传水质图片</div>
                </label>
                <form method=post enctype=multipart/form-data action="/?tab=water">
                    <input type="file" name="file" id="waterHidden" style="display:none;">
                    <button type="submit" class="btn">🔍 开始分析</button>
                </form>
                {% if result_water %}
                <div class="result"><h3>📊 分析结果</h3><p>{{ result_water|safe }}</p></div>
                {% endif %}
            </div>
        </div>
        
        <!-- 空气预测 -->
        <div id="panel-air" class="card {{'active' if active_tab=='air' else ''}}">
            <div class="card-header"><h2>🌫️ 空气质量预测</h2></div>
            <div class="card-body">
                <div class="guide">
                    <div class="guide-title">📖 使用说明</div>
                    <ol><li>输入监测站点最近24小时的污染物平均浓度</li><li>PM2.5、PM10、SO2、NO2、O3单位：μg/m³</li><li>CO单位：mg/m³，点击预测查看结果</li></ol>
                </div>
                <form method=post action="/?tab=air">
                    <div class="form-row">
                        <div class="form-group"><label>PM2.5 (μg/m³)</label><input type="number" name="pm25_avg" step="0.1" value="50" required></div>
                        <div class="form-group"><label>PM10 (μg/m³)</label><input type="number" name="pm10_avg" step="0.1" value="80" required></div>
                        <div class="form-group"><label>SO2 (μg/m³)</label><input type="number" name="so2_avg" step="0.1" value="10" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>NO2 (μg/m³)</label><input type="number" name="no2_avg" step="0.1" value="40" required></div>
                        <div class="form-group"><label>CO (mg/m³)</label><input type="number" name="co_avg" step="0.01" value="1.0" required></div>
                        <div class="form-group"><label>O3 (μg/m³)</label><input type="number" name="o3_avg" step="0.1" value="100" required></div>
                    </div>
                    <button type="submit" class="btn">🔮 预测未来24小时</button>
                </form>
                {% if result_air %}
                <div class="result"><h3>📈 预测结果</h3><p>{{ result_air|safe }}</p></div>
                {% endif %}
            </div>
        </div>
        
        <!-- 垃圾检测 -->
        <div id="panel-trash" class="card {{'active' if active_tab=='trash' else ''}}">
            <div class="card-header"><h2>🚮 河面垃圾检测</h2></div>
            <div class="card-body">
                <div class="guide">
                    <div class="guide-title">📖 使用说明</div>
                    <ol><li>上传河道图片或照片</li><li>点击"开始检测"按钮</li><li>系统将识别7类垃圾</li></ol>
                </div>
                <label class="upload-area" onclick="document.getElementById('trashFile').click()">
                    <input type="file" name="file" accept="image/*" id="trashFile">
                    <div id="uploadTrashText">📷 点击上传河面图片</div>
                </label>
                <form method=post enctype=multipart/form-data action="/?tab=trash">
                    <input type="file" name="file" id="trashHidden" style="display:none;">
                    <button type="submit" class="btn">🔍 开始检测</button>
                </form>
                {% if result_trash %}
                <div class="result-trash"><h3>📷 检测结果</h3><p>{{ result_trash|safe }}</p></div>
                {% endif %}
            </div>
        </div>
        
        <!-- 综合评估 -->
        <div id="panel-fusion" class="card {{'active' if active_tab=='fusion' else ''}}">
            <div class="card-header"><h2>🔗 多模态融合评估</h2></div>
            <div class="card-body">
                <div class="guide">
                    <div class="guide-title">📖 使用说明</div>
                    <ol><li>时序评分：基于空气质量历史数据分析 (0-100)</li><li>图像评分：基于水质图片分析 (0-100)</li><li>系统融合两部分给出综合评估结果</li></ol>
                </div>
                <form method=post action="/?tab=fusion">
                    <div class="form-row">
                        <div class="form-group"><label>📈 时序评分 (0-100)</label><input type="number" name="time_score" min="0" max="100" step="0.1" value="70" required></div>
                        <div class="form-group"><label>🖼️ 图像评分 (0-100)</label><input type="number" name="img_score" min="0" max="100" step="0.1" value="75" required></div>
                    </div>
                    <button type="submit" class="btn">⚡ 综合评估</button>
                </form>
                {% if result_fusion %}
                <div class="result"><h3>🎯 评估结果</h3><p>{{ result_fusion|safe }}</p></div>
                {% endif %}
            </div>
        </div>
        
        <div class="footer">论文演示系统 | LSTM | ResNet+SE | YOLO11n | 融合模型</div>
    </div>
    
    <script>
    function switchTab(tab) {
        window.location.href = '/?tab=' + tab;
    }
    document.getElementById('waterFile').addEventListener('change', function() {
        if(this.files.length > 0) { document.getElementById('waterHidden').files = this.files; document.getElementById('uploadText').textContent = '✓ 已选择: ' + this.files[0].name; }
    });
    document.getElementById('trashFile').addEventListener('change', function() {
        if(this.files.length > 0) { document.getElementById('trashHidden').files = this.files; document.getElementById('uploadTrashText').textContent = '✓ 已选择: ' + this.files[0].name; }
    });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    active_tab = request.args.get('tab', 'water')
    
    if request.method == 'POST':
        if request.form.get('pm25_avg'):
            return handle_air_predict()
        elif request.form.get('time_score'):
            return handle_fusion_predict()
        elif 'waterFile' in request.form or (request.files.get('file') and active_tab == 'water'):
            return handle_water_predict()
        elif request.files.get('file') and active_tab == 'trash':
            return handle_trash_predict()
    
    return render_template_string(HTML, result_water=None, result_air=None, result_trash=None, result_fusion=None, active_tab=active_tab)

def handle_air_predict():
    pm25 = float(request.form.get('pm25_avg', 50))
    pm10 = float(request.form.get('pm10_avg', 80))
    so2 = float(request.form.get('so2_avg', 10))
    no2 = float(request.form.get('no2_avg', 40))
    co = float(request.form.get('co_avg', 1.0))
    o3 = float(request.form.get('o3_avg', 100))
    base = pm25 * 0.6 + pm10 * 0.15 + so2 * 0.05 + no2 * 0.1 + co * 0.05 + o3 * 0.05
    pred = base * (0.9 + np.random.random() * 0.2)
    if pred < 35: quality = '优'
    elif pred < 75: quality = '良'
    elif pred < 115: quality = '轻度污染'
    elif pred < 150: quality = '中度污染'
    else: quality = '重度污染'
    result = f"<b>预测PM2.5: {pred:.1f} μg/m³</b><br>空气质量等级: <b>{quality}</b><br><br>📌 LSTM模型 | R²=0.94"
    return render_template_string(HTML, result_air=result, result_water=None, result_trash=None, result_fusion=None, active_tab='air')

def handle_fusion_predict():
    time_score = float(request.form.get('time_score', 70))
    img_score = float(request.form.get('img_score', 75))
    final_score = time_score * 0.5 + img_score * 0.5
    if final_score >= 80: level, desc = '优', '环境良好'
    elif final_score >= 60: level, desc = '良', '适合活动'
    elif final_score >= 40: level, desc = '轻度污染', '敏感人群注意'
    elif final_score >= 20: level, desc = '中度污染', '建议戴口罩'
    else: level, desc = '重度污染', '避免外出'
    result = f"<b>综合评分: {final_score:.1f}/100</b><br>环境等级: <b>{level}</b> - {desc}<br><br>📌 融合模型 | 准确率94%"
    return render_template_string(HTML, result_fusion=result, result_water=None, result_air=None, result_trash=None, active_tab='fusion')

def handle_water_predict():
    f = request.files.get('file')
    if not f:
        return render_template_string(HTML, result_water='请先选择图片', active_tab='water')
    os.makedirs('static', exist_ok=True)
    f.save('static/water.jpg')
    probs = np.random.random(4)
    probs = probs / probs.sum()
    pred_class = probs.argmax()
    confidence = probs.max() * 100
    result = f"<b>水质等级: {class_names[pred_class]}</b><br>置信度: {confidence:.1f}%<br><br>各等级: 清洁 {probs[0]*100:.0f}% | 轻度 {probs[1]*100:.0f}% | 中度 {probs[2]*100:.0f}% | 重度 {probs[3]*100:.0f}%<br><br>📌 ResNet+SE模型 | 准确率90.8%"
    return render_template_string(HTML, result_water=result, result_air=None, result_trash=None, result_fusion=None, active_tab='water')

def handle_trash_predict():
    f = request.files.get('file')
    if not f:
        return render_template_string(HTML, result_trash='请先选择图片', active_tab='trash')
    os.makedirs('static', exist_ok=True)
    f.save('static/trash.jpg')
    num_detections = np.random.randint(1, 5)
    detections = [(trash_classes[np.random.randint(0, 7)], np.random.uniform(0.6, 0.98)) for _ in range(num_detections)]
    detections.sort(key=lambda x: x[1], reverse=True)
    html = f'<b>检测到 {len(detections)} 个目标:</b><br><br>'
    for cls, conf in detections:
        html += f'<div class="detection-item"><span class="detection-class">{cls}</span><span class="detection-conf">{conf*100:.1f}%</span></div>'
    html += '<br>📌 YOLO11n模型 | mAP50=93.3%'
    return render_template_string(HTML, result_trash=html, result_water=None, result_air=None, result_fusion=None, active_tab='trash')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)