import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import serve_score  

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_and_extract(video_path):
    """提取骨骼数据的同时，把带火柴人的视频画出来"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not np.isfinite(fps): fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 准备写入带骨骼的视频
    out_path = "output_drawn.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, int(fps), (width, height))
    
    data = []
    frame_idx = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            ts_ms = frame_idx * (1000.0 / fps)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # 绘制绿色火柴人骨骼
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # 提取数据存入 DataFrame
                for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                    data.append({
                        "frame": frame_idx,
                        "ts_ms": ts_ms,
                        "lm": lm_id,
                        # 云端精简版暂用原始坐标代替平滑坐标，依赖 serve_score 内部的中值滤波
                        "smooth_x": lm.x,  
                        "smooth_y": lm.y,
                        "vis": lm.visibility
                    })
            
            # 可以在左上角加个帧号，方便你排查哪一帧出框了
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            out.write(frame)
            frame_idx += 1
            
    cap.release()
    out.release()
    return pd.DataFrame(data), out_path

def analyze_serve_cloud(video_path):
    if not video_path: return None, "请先上传视频。"
    
    try:
        # 1. 跑 MediaPipe 提取数据并渲染机甲视频
        df, out_video_path = process_video_and_extract(video_path)
        if df.empty:
            return None, "未能检测到人体骨骼，请更换视频。"
            
        # 2. 调用核心算法引擎
        result = serve_score.analyze_serve(df, use_coords="smooth")
        
        # 3. 解析报告
        final_score = result.get("final_score", 0)
        subscores = result.get("subscores", {})
        metrics = result.get("metrics", {})
        phases = result.get("phases", {})
        
        report = f"### 🏆 综合发球评分: {final_score:.1f}/100\n\n"
        
        report += "#### 📊 核心指标打分\n"
        report += f"- **下肢驱动 (Knee Bend)**: {subscores.get('knee_flexion', 0):.1f} 分\n"
        report += f"- **躯干控制 (Trunk Inclination)**: {subscores.get('trunk_inclination', 0):.1f} 分\n"
        report += f"- **击球高度 (Impact Height)**: {subscores.get('impact_height', 0):.1f} 分\n"
        report += f"- **发力节奏 (Timing)**: {subscores.get('timing', 0):.1f} 分\n\n"
        
        report += "#### ⏱️ 关键动作帧位 (Phase Detection)\n"
        report += f"- 抛球最高点: 第 {phases.get('toss_peak', 'N/A')} 帧\n"
        report += f"- 奖杯姿势 (HRP): 第 {phases.get('trophy', 'N/A')} 帧\n"
        report += f"- 拍头最低点 (LRP): 第 {phases.get('racket_drop', 'N/A')} 帧\n"
        report += f"- 击球瞬间: 第 {phases.get('contact', 'N/A')} 帧\n\n"
        
        report += "#### 🔬 生物力学量化数据\n"
        report += f"- 估算膝盖弯曲度: {metrics.get('knee_flexion_est_deg', 0):.1f}°\n"
        report += f"- 估算躯干倾斜度: {metrics.get('trunk_inclination_est_deg', 0):.1f}°\n"
        
        impact_ratio = metrics.get('impact_bh_ratio', 0)
        report += f"- 击球点/身高比值: {impact_ratio:.2f}x "
        if not np.isfinite(impact_ratio):
            report += "*(⚠️ 骨骼出框或遮挡，请核对右侧回放视频的击球帧)*"
            
        return out_video_path, report
        
    except Exception as e:
        return None, f"分析出错: {str(e)}"

# 构建手机端极简 UI
with gr.Blocks(title="Serve Score Pro") as demo:
    gr.Markdown("## 🎾 职业发球生物力学分析 (云端诊断版)")
    
    with gr.Row():
        video_input = gr.Video(label="上传网球发球视频")
    submit_btn = gr.Button("🚀 运行分析引擎", variant="primary")
    
    with gr.Row():
        video_output = gr.Video(label="AI 骨骼追踪回放 (用于人工核对出框/跳点)")
        text_output = gr.Markdown(label="评估报告")
        
    submit_btn.click(fn=analyze_serve_cloud, inputs=video_input, outputs=[video_output, text_output])

if __name__ == "__main__":
    demo.launch()