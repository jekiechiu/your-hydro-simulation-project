import io
import json
import numpy as np
import pandas as pd
from flask import Flask, request, send_from_directory, render_template_string
import os
from scipy.ndimage import zoom # 新增，用於地形數據擴展

app = Flask(__name__)

# --- 水文模擬核心邏輯 (使用您提供的邏輯，並修正降雨量計算) ---
def run_water_simulation(original_height_map_raw, dx, dy, rainfall_type_code, simulation_duration_steps):
    
    # 1. 地形數據預處理：將原始地形擴展
    # 如果原始數據太小（例如1xN或Nx1），scipy.ndimage.zoom可能會導致錯誤或行為不符合預期
    # 這裡增加一個判斷，避免在不適合插值的情況下進行插值。
    # 至少需要 2x2 的網格才能進行 order=3 的插值，否則邊界處理會有問題。
    if original_height_map_raw.shape[0] > 1 and original_height_map_raw.shape[1] > 1:
        # 您的 Tkinter 代碼中的 zoom_factors 計算：
        # ((data.shape[0] - 1) * 10 + 1) / (data.shape[0] - 1)
        # 這意味著如果原始有 N 個間隔，會變成 (N-1)*10 個間隔
        # 也就是每個原始間隔被分成了約 10 個小間隔
        zoom_level_x = ((original_height_map_raw.shape[1] - 1) * 10 + 1) / (original_height_map_raw.shape[1] - 1) if original_height_map_raw.shape[1] > 1 else 1.0
        zoom_level_y = ((original_height_map_raw.shape[0] - 1) * 10 + 1) / (original_height_map_raw.shape[0] - 1) if original_height_map_raw.shape[0] > 1 else 1.0
        
        expanded_height_map = zoom(original_height_map_raw, zoom=(zoom_level_y, zoom_level_x), order=3)
        
        # 我們需要一個 "有效的" 縮放因子來計算每個小網格的降雨量
        # 這個因子指的是原始一個網格單元在每個維度上被分成了多少個小網格單元
        # 這裡假設您的意圖是將每個原始單元分割成約 10x10 個子單元
        effective_subdivision_factor = 10.0 # 每個原始間隔細化為 10 個小間隔
    else:
        # 如果原始數據是 1xN, Nx1 或 1x1，不進行插值
        expanded_height_map = original_height_map_raw
        effective_subdivision_factor = 1.0 # 沒有細化

    rows, cols = expanded_height_map.shape
    
    # 根據台灣雨量規範定義的 24 小時總降雨量 (單位: mm)
    rainfall_mm_per_24hr = {
        "A": 80.0,  # 大雨
        "B": 200.0, # 豪雨
        "C": 350.0, # 豪大雨
        "D": 500.0  # 超大豪雨
    }
    
    # 獲取選定的 24 小時總降雨量 (mm)
    selected_rainfall_mm_per_24hr = rainfall_mm_per_24hr.get(rainfall_type_code, rainfall_mm_per_24hr["A"])
    
    # 將 mm/24hr 轉換為 m/min
    # 1 mm = 0.001 m
    # 1 天 = 24 小時 = 1440 分鐘
    # 這是單位面積在單位時間內應增加的降雨高度 (m/min)
    base_rainfall_height_per_min = (selected_rainfall_mm_per_24hr / 1000.0) / 1440.0 

    # 計算細化後每個小網格點每分鐘實際接收的降雨深度
    # 由於總降雨量應保持不變，且地形被細化了，所以每個小網格分攤到的雨量要減少。
    # 如果原始的一個單元 (dx*dy) 被分成了 (effective_subdivision_factor * effective_subdivision_factor) 個小單元
    # 那麼每個小單元在單位時間內獲得的雨量就應該是 `base_rainfall_height_per_min` 除以 `effective_subdivision_factor^2`
    rainfall_per_expanded_grid_cell_per_min = base_rainfall_height_per_min / (effective_subdivision_factor ** 2)

    # 初始化用於儲存結果的字典
    # current_state_map 將包含地形高程和積水深度之和
    simulation_results_map = {"t=0": expanded_height_map.copy()}
    
    # 定義鄰居關係
    neighbors_dict = {
        (i, j): [
            (i + di, j + dj)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] # 四個方向
            if 0 <= i + di < rows and 0 <= j + dj < cols
        ]
        for i in range(rows)
        for j in range(cols)
    }

    # 模擬主迴圈
    current_state_map = expanded_height_map.copy() # 這是每個時間步的"總高度" (地形 + 累積水深)
    
    for t_step in range(1, simulation_duration_steps + 1):
        # 每個時間步，所有網格點都接收新的雨量
        # 這個 rain_input_map 是每個點在該步新接收的、用於流動和累積的"水量"
        rain_input_map = np.full((rows, cols), rainfall_per_expanded_grid_cell_per_min, dtype=float)

        # 遍歷從高到低的點，進行水流分配
        # 排序是基於 "當前總高度" (地形+累積水深)
        flat_indices = np.argsort(-current_state_map, axis=None) # 降序排列
        sorted_coords = np.unravel_index(flat_indices, current_state_map.shape)

        new_rain_distribution_map = rain_input_map.copy() # 記錄水流動後的雨量分佈

        for r_idx, c_idx in zip(*sorted_coords):
            current_total_height = current_state_map[r_idx, c_idx]
            
            # 找到比當前點 "水面總高" 更低的鄰居
            lower_neighbors = []
            for nr, nc in neighbors_dict[(r_idx, c_idx)]:
                if current_state_map[nr, nc] < current_total_height:
                    lower_neighbors.append((nr, nc))

            if lower_neighbors:
                # 找到最低的鄰居們（可能有多個）
                min_neighbor_height = min(current_state_map[nr, nc] for nr, nc in lower_neighbors)
                lowest_points = [
                    (nr, nc) for nr, nc in lower_neighbors
                    if current_state_map[nr, nc] == min_neighbor_height
                ]

                # 將當前點的 "新接收雨量" 分配給這些最低鄰居
                if len(lowest_points) > 0: # 避免除以零
                    flow_value = new_rain_distribution_map[r_idx, c_idx] / len(lowest_points)
                    for nr, nc in lowest_points:
                        new_rain_distribution_map[nr, nc] += flow_value
                    new_rain_distribution_map[r_idx, c_idx] = 0 # 當前點的雨量已流出
        
        # 更新當前時間步的總高度：前一時間步的總高度 + 經過流動分配後的新雨量
        current_state_map = current_state_map + new_rain_distribution_map 
        
        # 將當前時間步的結果存儲下來
        simulation_results_map[f"t={t_step}"] = current_state_map.copy()

    # 將所有時間步的結果轉換為列表的列表，以便 JSON 傳輸
    # 我們需要返回擴展後的原始地形和每個時間步的"水深"
    # 水深 = 總高度 - 原始地形高度
    simulation_snapshots_for_frontend = []
    initial_terrain_for_frontend = expanded_height_map.tolist() # 擴展後的地形

    for t_key in sorted(simulation_results_map.keys(), key=lambda x: int(x.split('=')[1])):
        total_height_at_t = np.array(simulation_results_map[t_key])
        # 計算實際的水深 (總高度 - 原始地形高度)
        # 如果結果小於原始地形高度，表示沒有水，水深為0
        water_depth_at_t = np.maximum(0, total_height_at_t - expanded_height_map)
        simulation_snapshots_for_frontend.append(water_depth_at_t.tolist())
    
    return initial_terrain_for_frontend, simulation_snapshots_for_frontend


# 全局變數用於儲存地形和模擬結果
global_original_terrain = None # 擴展後的原始地形，用於前端顯示灰色網格
global_simulation_snapshots = None # 每個時間步的水深數據
global_sim_current_step = 0

# --- 路由定義 ---

# 提供前端 HTML 檔案
@app.route('/')
def serve_index():
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

# 處理 Excel 檔案上傳和模擬參數
@app.route('/upload-and-simulate', methods=['POST'])
def upload_and_simulate():
    global global_original_terrain, global_simulation_snapshots, global_sim_current_step

    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    if file and file.filename.endswith(('.xlsx', '.xls')): # 支援 .xls
        try:
            df = pd.read_excel(io.BytesIO(file.read()), header=None) # 假設沒有header
            original_height_map_raw = df.to_numpy()
            
            # 從表單數據獲取模擬參數
            dx = float(request.form.get('dx', 10.0)) # 預設值 10m
            dy = float(request.form.get('dy', 10.0))
            rainfall_type = request.form.get('rainfall_type', 'A') # 預設 A
            simulation_time_min = float(request.form.get('sim_time', 60.0)) # 分鐘
            
            # 運行水流模擬
            global_original_terrain, global_simulation_snapshots = run_water_simulation(
                original_height_map_raw, dx, dy, rainfall_type, int(simulation_time_min)
            )
            global_sim_current_step = 0 # 重置模擬步長

            rows_expanded = len(global_original_terrain)
            cols_expanded = len(global_original_terrain[0]) if rows_expanded > 0 else 0

            return {
                "status": "success",
                "message": "Simulation started",
                "terrain_data": global_original_terrain, # 這裡返回的是擴展後的地形
                "rows": rows_expanded,
                "cols": cols_expanded,
                "dx": dx, # 這些 dx, dy 用於前端 Three.js 空間縮放
                "dy": dy,
                "total_steps": len(global_simulation_snapshots)
            }
        except Exception as e:
            # 捕捉讀取Excel或模擬計算中的錯誤
            return {"error": f"Failed to process Excel or run simulation: {str(e)}"}, 500
    else:
        return {"error": "Invalid file type. Please upload an .xlsx or .xls file."}, 400

# 提供實時水深數據
@app.route('/get-water-data', methods=['GET'])
def get_water_data():
    global global_simulation_snapshots, global_sim_current_step

    if global_simulation_snapshots is None or not global_simulation_snapshots:
        return {"error": "No simulation running or no snapshots available"}, 400
    
    if global_sim_current_step < len(global_simulation_snapshots):
        water_data = global_simulation_snapshots[global_sim_current_step]
        current_step_display = global_sim_current_step # 用於顯示的步數
        global_sim_current_step += 1 # 為下次請求增加
        return {
            "water_depth": water_data,
            "current_step": current_step_display, # 返回當前讀取的步數
            "total_steps": len(global_simulation_snapshots)
        }
    else:
        return {"status": "simulation_finished", "message": "No more water data"}

if __name__ == '__main__':
    app.run(debug=True)
