from graphviz import Digraph

# --- 辅助函数定义 (从上方复制代码到这里) ---
def create_stage_graph(filename_prefix, stage_id, stage_title, internal_nodes_defs, internal_edges_defs,
                       input_connections=None, output_connections=None,
                       default_node_font='Microsoft YaHei', default_edge_font='Microsoft YaHei',
                       default_node_color='lightgreen', connector_node_color='lightgray',
                       connector_shape='ellipse', main_node_shape='box', rankdir='LR'):
    """
    创建并渲染一个特定阶段的流程图。

    Args:
        filename_prefix (str): 生成图片文件名的前缀。
        stage_id (str): 阶段的唯一标识符，用于文件名。
        stage_title (str): 图的标题，也会显示在图的顶部。
        internal_nodes_defs (dict): 阶段内部节点的定义 {node_id: label_text}。
        internal_edges_defs (list): 阶段内部边的定义，每个元素为元组
                                   (source_id, target_id, optional_attributes_dict)。
                                   例如: [('A', 'B'), ('B', 'C', {'label': '详情'})]
        input_connections (dict, optional): 输入连接定义
                                           {internal_node_id_as_target: external_source_description}。
                                           例如: {'NODE_IN_STAGE_A': '来自: 前置准备阶段'}
        output_connections (dict, optional): 输出连接定义
                                            {internal_node_id_as_source: external_target_description}。
                                            例如: {'NODE_IN_STAGE_Z': '去往: 后续处理阶段'}
        default_node_font (str): 默认节点字体。
        default_edge_font (str): 默认边字体。
        default_node_color (str): 默认节点颜色。
        connector_node_color (str): 连接器节点颜色。
        connector_shape (str): 连接器节点形状。
        main_node_shape (str): 主要流程节点形状。
        rankdir (str): 布局方向 ('LR' 或 'TB')。
    """
    dot = Digraph(name=f"cluster_{stage_id}", comment=stage_title) 
    dot.attr(rankdir=rankdir, label=stage_title, fontname=default_node_font, fontsize='16', labelloc='t')
    dot.attr('node', shape=main_node_shape, style='filled', color=default_node_color, fontname=default_node_font)
    dot.attr('edge', fontname=default_edge_font)

    if input_connections:
        for target_node_id, source_desc in input_connections.items():
            connector_id = f"input_conn_{stage_id}_{target_node_id.replace(':', '_')}" # Sanitize ID
            dot.node(connector_id, source_desc, shape=connector_shape, style='filled', color=connector_node_color)
            if target_node_id in internal_nodes_defs:
                dot.edge(connector_id, target_node_id)

    for node_id, label in internal_nodes_defs.items():
        dot.node(node_id, label)

    for edge_def in internal_edges_defs:
        u, v = edge_def[0], edge_def[1]
        attrs = edge_def[2] if len(edge_def) > 2 else {}
        if u in internal_nodes_defs and v in internal_nodes_defs:
             dot.edge(u, v, **attrs)

    if output_connections:
        for source_node_id, target_desc in output_connections.items():
            connector_id = f"output_conn_{stage_id}_{source_node_id.replace(':', '_')}" # Sanitize ID
            dot.node(connector_id, target_desc, shape=connector_shape, style='filled', color=connector_node_color)
            if source_node_id in internal_nodes_defs:
                dot.edge(source_node_id, connector_id)
    
    try:
        output_base = f"{filename_prefix}_{stage_id}"
        dot.render(output_base, view=False, cleanup=True, format='png')
        print(f"流程图阶段 '{stage_title}' 已生成: {output_base}.png")
    except Exception as e:
        print(f"生成流程图阶段 '{stage_title}' 失败 ({output_base}.png): {e}")

# --- 节点统一定义 (方便引用) ---
INFERENCE_NODES = {
    'START_INFER': '开始推理',
    'LOAD_ARGS_INFER': '加载推理参数\n(模型路径, 输入源等)',
    'INIT_MODEL_INFER': '初始化模型\n(PoseEstimationWithMobileNet)',
    'LOAD_TRAINED_MODEL': '加载训练好的模型参数\n(checkpoint_path)',
    'SET_DEVICE': '设置计算设备\n(CPU/GPU)',
    'PREPARE_INPUT': '准备输入源\n(ImageReader/VideoReader)',
    'INFER_LOOP': '推理循环 (逐帧/逐图)',
    'READ_FRAME': '读取图像/视频帧',
    'PREPROCESS_IMAGE': '图像预处理\n(Resize, Normalize, Pad)',
    'MODEL_INFER': '模型推理 (infer_fast)\n(获取Heatmaps, PAFs)',
    'POSTPROCESS_OUTPUT': '后处理输出',
    'EXTRACT_KEYPOINTS': '提取关键点\n(extract_keypoints)',
    'GROUP_KEYPOINTS': '组合关键点\n(group_keypoints)',
    'CONVERT_COORDINATES': '转换坐标',
    'CREATE_POSES': '创建姿态对象 (Pose)',
    'DRAW_POSES': '绘制姿态到图像',
    'DISPLAY_OUTPUT': '显示/保存结果图像/视频',
    'OPTIONAL_VISUALIZE_HEATMAP': '(可选)可视化Heatmap',
    'OPTIONAL_VISUALIZE_PAF': '(可选)可视化PAF',
    'END_INFER': '结束推理'
}

# --- 生成独立阶段图 ---
FN_PREFIX_INFER = 'inference'

# 阶段1: 设置与准备
stage1_nodes = {
    'LOAD_ARGS_INFER': INFERENCE_NODES['LOAD_ARGS_INFER'],
    'INIT_MODEL_INFER': INFERENCE_NODES['INIT_MODEL_INFER'],
    'LOAD_TRAINED_MODEL': INFERENCE_NODES['LOAD_TRAINED_MODEL'],
    'SET_DEVICE': INFERENCE_NODES['SET_DEVICE'],
    'PREPARE_INPUT': INFERENCE_NODES['PREPARE_INPUT']
}
stage1_edges = [
    ('LOAD_ARGS_INFER', 'INIT_MODEL_INFER'),
    ('INIT_MODEL_INFER', 'LOAD_TRAINED_MODEL'),
    ('LOAD_TRAINED_MODEL', 'SET_DEVICE'),
    ('SET_DEVICE', 'PREPARE_INPUT')
]
create_stage_graph(FN_PREFIX_INFER, '1_setup', '推理阶段一: 设置与准备',
                   stage1_nodes, stage1_edges,
                   input_connections={'LOAD_ARGS_INFER': '来自: 开始推理节点'},
                   output_connections={'PREPARE_INPUT': '去往: 推理循环'})

# 阶段2: 数据处理 (循环内)
stage2_nodes = {
    'READ_FRAME': INFERENCE_NODES['READ_FRAME'],
    'PREPROCESS_IMAGE': INFERENCE_NODES['PREPROCESS_IMAGE']
}
stage2_edges = [('READ_FRAME', 'PREPROCESS_IMAGE')]
create_stage_graph(FN_PREFIX_INFER, '2_data_processing', '推理阶段二: 数据处理 (循环内)',
                   stage2_nodes, stage2_edges,
                   input_connections={'READ_FRAME': '来自: 推理循环 (新帧/图)'},
                   output_connections={'PREPROCESS_IMAGE': '去往: 模型推理'})

# 阶段3: 模型推理 (循环内)
stage3_nodes = {'MODEL_INFER': INFERENCE_NODES['MODEL_INFER']}
stage3_edges = [] # 单节点无内部边
create_stage_graph(FN_PREFIX_INFER, '3_model_inference', '推理阶段三: 模型推理 (循环内)',
                   stage3_nodes, stage3_edges,
                   input_connections={'MODEL_INFER': '来自: 图像预处理'},
                   output_connections={'MODEL_INFER': '去往: 后处理 / 可选可视化'})

# 阶段4: 后处理 (循环内)
stage4_nodes = {
    'POSTPROCESS_OUTPUT': INFERENCE_NODES['POSTPROCESS_OUTPUT'],
    'EXTRACT_KEYPOINTS': INFERENCE_NODES['EXTRACT_KEYPOINTS'],
    'GROUP_KEYPOINTS': INFERENCE_NODES['GROUP_KEYPOINTS'],
    'CONVERT_COORDINATES': INFERENCE_NODES['CONVERT_COORDINATES'],
    'CREATE_POSES': INFERENCE_NODES['CREATE_POSES']
}
stage4_edges = [
    ('POSTPROCESS_OUTPUT', 'EXTRACT_KEYPOINTS'),
    ('EXTRACT_KEYPOINTS', 'GROUP_KEYPOINTS'),
    ('GROUP_KEYPOINTS', 'CONVERT_COORDINATES'),
    ('CONVERT_COORDINATES', 'CREATE_POSES')
]
create_stage_graph(FN_PREFIX_INFER, '4_postprocessing', '推理阶段四: 后处理 (循环内)',
                   stage4_nodes, stage4_edges,
                   input_connections={'POSTPROCESS_OUTPUT': '来自: 模型推理'},
                   output_connections={'CREATE_POSES': '去往: 绘制姿态'})

# 阶段5: 可视化与输出 (循环内)
stage5_nodes = {
    'DRAW_POSES': INFERENCE_NODES['DRAW_POSES'],
    'DISPLAY_OUTPUT': INFERENCE_NODES['DISPLAY_OUTPUT']
}
stage5_edges = [('DRAW_POSES', 'DISPLAY_OUTPUT')]
create_stage_graph(FN_PREFIX_INFER, '5_visualization_output', '推理阶段五: 可视化与输出 (循环内)',
                   stage5_nodes, stage5_edges,
                   input_connections={
                       'DRAW_POSES': '来自: 创建姿态对象',
                       'DISPLAY_OUTPUT': '来自: (可选)特征图可视化' # 简化连接器
                   },
                   output_connections={'DISPLAY_OUTPUT': '去往: 推理循环 (下一帧/图)'})

# 阶段6: 可选辅助可视化 (循环内)
stage6_nodes = {
    'OPTIONAL_VISUALIZE_HEATMAP': INFERENCE_NODES['OPTIONAL_VISUALIZE_HEATMAP'],
    'OPTIONAL_VISUALIZE_PAF': INFERENCE_NODES['OPTIONAL_VISUALIZE_PAF']
}
stage6_edges = [] # 通常是并行节点
create_stage_graph(FN_PREFIX_INFER, '6_optional_visualization', '推理阶段六: (可选)辅助特征图可视化',
                   stage6_nodes, stage6_edges,
                   input_connections={
                       'OPTIONAL_VISUALIZE_HEATMAP': '来自: 模型推理',
                       'OPTIONAL_VISUALIZE_PAF': '来自: 模型推理'
                   },
                   output_connections={
                       'OPTIONAL_VISUALIZE_HEATMAP': '去往: 显示/保存结果',
                       'OPTIONAL_VISUALIZE_PAF': '去往: 显示/保存结果'
                   })

# --- 生成主控流程图 ---
dot_control_infer = Digraph(comment='Inference Overall Control Flow', format='png')
dot_control_infer.attr(rankdir='LR', label='推理主控流程', fontname='Microsoft YaHei', fontsize='18', labelloc='t')
dot_control_infer.attr('node', shape='box3d', style='filled', color='lightblue', fontname='Microsoft YaHei') # Stages as boxes
dot_control_infer.attr('edge', fontname='Microsoft YaHei')

# 控制节点
dot_control_infer.node('START_INFER_CTRL', INFERENCE_NODES['START_INFER'], shape='ellipse', color='lightgreen')
dot_control_infer.node('INFER_LOOP_CTRL', INFERENCE_NODES['INFER_LOOP'], shape='diamond', color='yellow')
dot_control_infer.node('END_INFER_CTRL', INFERENCE_NODES['END_INFER'], shape='ellipse', color='lightgreen')

# 阶段代表节点
dot_control_infer.node('STAGE_1_SETUP', '阶段一:\n设置与准备')
dot_control_infer.node('STAGE_2_DATA_PROC', '阶段二:\n数据处理')
dot_control_infer.node('STAGE_3_MODEL_INFER', '阶段三:\n模型推理')
dot_control_infer.node('STAGE_4_POST_PROC', '阶段四:\n后处理')
dot_control_infer.node('STAGE_5_VIZ_OUT', '阶段五:\n可视化与输出')
dot_control_infer.node('STAGE_6_OPT_VIZ', '(可选)\n辅助可视化', color='lightgoldenrodyellow') # Different color for optional

# 连接
dot_control_infer.edge('START_INFER_CTRL', 'STAGE_1_SETUP')
dot_control_infer.edge('STAGE_1_SETUP', 'INFER_LOOP_CTRL', lhead='cluster_loop_entry') # Connect to loop cluster if possible

with dot_control_infer.subgraph(name='cluster_loop_entry') as loop_entry: # Group loop related items
    loop_entry.edge('INFER_LOOP_CTRL', 'STAGE_2_DATA_PROC', label='新帧/图')
    loop_entry.edge('STAGE_2_DATA_PROC', 'STAGE_3_MODEL_INFER')
    loop_entry.edge('STAGE_3_MODEL_INFER', 'STAGE_4_POST_PROC')
    loop_entry.edge('STAGE_3_MODEL_INFER', 'STAGE_6_OPT_VIZ', constraint='false', style='dashed')
    loop_entry.edge('STAGE_4_POST_PROC', 'STAGE_5_VIZ_OUT')
    loop_entry.edge('STAGE_6_OPT_VIZ', 'STAGE_5_VIZ_OUT', constraint='false', style='dashed')
    loop_entry.edge('STAGE_5_VIZ_OUT', 'INFER_LOOP_CTRL', label='下一帧/图')

dot_control_infer.edge('INFER_LOOP_CTRL', 'END_INFER_CTRL', label='处理完成')

try:
    dot_control_infer.render(f'{FN_PREFIX_INFER}_0_overall_control_flow', view=False, cleanup=True)
    print(f"推理主控流程图已生成: {FN_PREFIX_INFER}_0_overall_control_flow.png")
except Exception as e:
    print(f"生成推理主控流程图失败: {e}")