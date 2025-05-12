from graphviz import Digraph

# --- 辅助函数定义 (从上方复制代码到这里 或确保可导入) ---
def create_stage_graph(filename_prefix, stage_id, stage_title, internal_nodes_defs, internal_edges_defs,
                       input_connections=None, output_connections=None,
                       default_node_font='Microsoft YaHei', default_edge_font='Microsoft YaHei',
                       default_node_color='skyblue', connector_node_color='lightgray', # Adjusted default color
                       connector_shape='ellipse', main_node_shape='box', rankdir='LR'):
    """
    创建并渲染一个特定阶段的流程图。
    (函数体与 inference_flow.py 中的版本相同)
    """
    dot = Digraph(name=f"cluster_{stage_id}", comment=stage_title) 
    dot.attr(rankdir=rankdir, label=stage_title, fontname=default_node_font, fontsize='16', labelloc='t')
    dot.attr('node', shape=main_node_shape, style='filled', color=default_node_color, fontname=default_node_font)
    dot.attr('edge', fontname=default_edge_font)

    if input_connections:
        for target_node_id, source_desc in input_connections.items():
            connector_id = f"input_conn_{stage_id}_{target_node_id.replace(':', '_')}"
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
            connector_id = f"output_conn_{stage_id}_{source_node_id.replace(':', '_')}"
            dot.node(connector_id, target_desc, shape=connector_shape, style='filled', color=connector_node_color)
            if source_node_id in internal_nodes_defs:
                dot.edge(source_node_id, connector_id)
    
    try:
        output_base = f"{filename_prefix}_{stage_id}"
        dot.render(output_base, view=False, cleanup=True, format='png')
        print(f"流程图阶段 '{stage_title}' 已生成: {output_base}.png")
    except Exception as e:
        print(f"生成流程图阶段 '{stage_title}' 失败 ({output_base}.png): {e}")


# --- 节点统一定义 ---
TRAINING_NODES = {
    'START_TRAIN': '开始训练',
    'LOAD_CONFIG': '加载配置参数\n(学习率, Batch Size, 迭代次数等)',
    'PREPARE_LABELS': '准备训练标签\n(prepare_train_labels.py)',
    'LOAD_DATASET': '加载数据集\n(CocoTrainDataset)',
    'DATA_TRANSFORM': '数据增强\n(旋转, 缩放, 裁剪, 翻转等)',
    'CREATE_DATALOADER': '创建数据加载器\n(DataLoader)',
    'INIT_MODEL': '初始化模型\n(PoseEstimationWithMobileNet)',
    'LOAD_CHECKPOINT': '(可选)加载预训练模型/检查点',
    'DEFINE_OPTIMIZER': '定义优化器\n(Adam)',
    'DEFINE_SCHEDULER': '定义学习率调度器',
    'TRAIN_LOOP': '训练循环 (Epochs)',
    'FORWARD_PASS': '前向传播',
    'CALCULATE_LOSS': '计算损失\n(Heatmaps Loss, PAFs Loss)',
    'BACKWARD_PASS': '反向传播',
    'OPTIMIZER_STEP': '更新模型参数',
    'SCHEDULER_STEP': '更新学习率',
    'LOG_LOSS': '记录损失',
    'SAVE_CHECKPOINT': '保存检查点',
    'VALIDATION': '(可选)执行验证\n(evaluate)',
    'END_TRAIN': '结束训练'
}

# --- 生成独立阶段图 ---
FN_PREFIX_TRAIN = 'training'

# 阶段1: 配置与数据准备
stage1_nodes_tr = {
    'LOAD_CONFIG': TRAINING_NODES['LOAD_CONFIG'],
    'PREPARE_LABELS': TRAINING_NODES['PREPARE_LABELS'],
    'LOAD_DATASET': TRAINING_NODES['LOAD_DATASET'],
    'DATA_TRANSFORM': TRAINING_NODES['DATA_TRANSFORM'],
    'CREATE_DATALOADER': TRAINING_NODES['CREATE_DATALOADER']
}
stage1_edges_tr = [
    ('LOAD_CONFIG', 'PREPARE_LABELS'),
    ('PREPARE_LABELS', 'LOAD_DATASET'),
    ('LOAD_DATASET', 'DATA_TRANSFORM'),
    ('DATA_TRANSFORM', 'CREATE_DATALOADER')
]
create_stage_graph(FN_PREFIX_TRAIN, '1_data_setup', '训练阶段一: 配置与数据准备',
                   stage1_nodes_tr, stage1_edges_tr, default_node_color='skyblue',
                   input_connections={'LOAD_CONFIG': '来自: 开始训练节点'},
                   output_connections={'CREATE_DATALOADER': '去往: 模型与优化器设置'})

# 阶段2: 模型与优化器设置
stage2_nodes_tr = {
    'INIT_MODEL': TRAINING_NODES['INIT_MODEL'],
    'LOAD_CHECKPOINT': TRAINING_NODES['LOAD_CHECKPOINT'],
    'DEFINE_OPTIMIZER': TRAINING_NODES['DEFINE_OPTIMIZER'],
    'DEFINE_SCHEDULER': TRAINING_NODES['DEFINE_SCHEDULER']
}
stage2_edges_tr = [
    ('INIT_MODEL', 'LOAD_CHECKPOINT'),
    ('LOAD_CHECKPOINT', 'DEFINE_OPTIMIZER'),
    ('DEFINE_OPTIMIZER', 'DEFINE_SCHEDULER')
]
create_stage_graph(FN_PREFIX_TRAIN, '2_model_optimizer_setup', '训练阶段二: 模型与优化器设置',
                   stage2_nodes_tr, stage2_edges_tr, default_node_color='skyblue',
                   input_connections={'INIT_MODEL': '来自: 数据加载器创建'},
                   output_connections={'DEFINE_SCHEDULER': '去往: 训练循环'})

# 阶段3: 核心训练步骤 (Epoch内)
stage3_nodes_tr = {
    'FORWARD_PASS': TRAINING_NODES['FORWARD_PASS'],
    'CALCULATE_LOSS': TRAINING_NODES['CALCULATE_LOSS'],
    'BACKWARD_PASS': TRAINING_NODES['BACKWARD_PASS'],
    'OPTIMIZER_STEP': TRAINING_NODES['OPTIMIZER_STEP'],
    'SCHEDULER_STEP': TRAINING_NODES['SCHEDULER_STEP']
}
stage3_edges_tr = [
    ('FORWARD_PASS', 'CALCULATE_LOSS'),
    ('CALCULATE_LOSS', 'BACKWARD_PASS'),
    ('BACKWARD_PASS', 'OPTIMIZER_STEP'),
    ('OPTIMIZER_STEP', 'SCHEDULER_STEP')
]
create_stage_graph(FN_PREFIX_TRAIN, '3_core_training_steps', '训练阶段三: 核心训练步骤 (Epoch内)',
                   stage3_nodes_tr, stage3_edges_tr, default_node_color='skyblue',
                   input_connections={'FORWARD_PASS': '来自: 训练循环 (新Epoch/Batch)'},
                   output_connections={'SCHEDULER_STEP': '去往: 记录、保存与验证'})

# 阶段4: 记录、保存与验证 (Epoch后)
stage4_nodes_tr = {
    'LOG_LOSS': TRAINING_NODES['LOG_LOSS'],
    'SAVE_CHECKPOINT': TRAINING_NODES['SAVE_CHECKPOINT'],
    'VALIDATION': TRAINING_NODES['VALIDATION']
}
stage4_edges_tr = [
    ('LOG_LOSS', 'SAVE_CHECKPOINT'),
    ('SAVE_CHECKPOINT', 'VALIDATION')
]
create_stage_graph(FN_PREFIX_TRAIN, '4_log_save_validate', '训练阶段四: 记录、保存与验证 (Epoch后)',
                   stage4_nodes_tr, stage4_edges_tr, default_node_color='skyblue',
                   input_connections={'LOG_LOSS': '来自: 学习率更新后'},
                   output_connections={'VALIDATION': '去往: 训练循环 (下一Epoch)'})


# --- 生成主控流程图 ---
dot_control_train = Digraph(comment='Training Overall Control Flow', format='png')
dot_control_train.attr(rankdir='LR', label='训练主控流程', fontname='Microsoft YaHei', fontsize='18', labelloc='t')
dot_control_train.attr('node', shape='box3d', style='filled', color='powderblue', fontname='Microsoft YaHei') # Stages as boxes
dot_control_train.attr('edge', fontname='Microsoft YaHei')

# 控制节点
dot_control_train.node('START_TRAIN_CTRL', TRAINING_NODES['START_TRAIN'], shape='ellipse', color='lightskyblue')
dot_control_train.node('TRAIN_LOOP_CTRL', TRAINING_NODES['TRAIN_LOOP'], shape='diamond', color='yellow')
dot_control_train.node('END_TRAIN_CTRL', TRAINING_NODES['END_TRAIN'], shape='ellipse', color='lightskyblue')

# 阶段代表节点
dot_control_train.node('STAGE_TR_1_DATA_SETUP', '阶段一:\n配置与数据准备')
dot_control_train.node('STAGE_TR_2_MODEL_OPT', '阶段二:\n模型与优化器设置')
dot_control_train.node('STAGE_TR_3_CORE_TRAIN', '阶段三:\n核心训练步骤')
dot_control_train.node('STAGE_TR_4_LOG_SAVE', '阶段四:\n记录、保存与验证')

# 连接
dot_control_train.edge('START_TRAIN_CTRL', 'STAGE_TR_1_DATA_SETUP')
dot_control_train.edge('STAGE_TR_1_DATA_SETUP', 'STAGE_TR_2_MODEL_OPT')
dot_control_train.edge('STAGE_TR_2_MODEL_OPT', 'TRAIN_LOOP_CTRL')

with dot_control_train.subgraph(name='cluster_train_loop_body') as loop_body:
    loop_body.edge('TRAIN_LOOP_CTRL', 'STAGE_TR_3_CORE_TRAIN', label='新Epoch/Batch')
    loop_body.edge('STAGE_TR_3_CORE_TRAIN', 'STAGE_TR_4_LOG_SAVE')
    loop_body.edge('STAGE_TR_4_LOG_SAVE', 'TRAIN_LOOP_CTRL', label='下一Epoch')

dot_control_train.edge('TRAIN_LOOP_CTRL', 'END_TRAIN_CTRL', label='训练完成')

try:
    dot_control_train.render(f'{FN_PREFIX_TRAIN}_0_overall_control_flow', view=False, cleanup=True)
    print(f"训练主控流程图已生成: {FN_PREFIX_TRAIN}_0_overall_control_flow.png")
except Exception as e:
    print(f"生成训练主控流程图失败: {e}")