from graphviz import Digraph

def generate_network_diagram(output_filename="your_network_architecture_diagram_zh",
                             num_refinement_stages=1, num_channels=128,
                             num_heatmaps=19, num_pafs=38,
                             font_name="Microsoft YaHei"):
    dot = Digraph(comment='PoseEstimationWithMobileNet 架构', format='png')
    dot.attr(rankdir='TB', label='PoseEstimationWithMobileNet 网络架构图', fontsize='20', labelloc='t', fontname=font_name)
    dot.attr('node', shape='box', style='filled', fontname=font_name)
    dot.attr('edge', fontname=font_name)

    # 输入节点
    dot.node('Input', '输入图像 (3, H, W)', color='lightgray')

    # 特征提取主干 (self.model)
    with dot.subgraph(name='cluster_backbone') as bb:
        bb.attr(label='特征提取主干 (MobileNet-like)', style='filled', color='lightblue', fontname=font_name)
        bb_nodes = [
            ('conv_3_32', '卷积(3, 32, s=2)'),
            ('dw_32_64', '深度卷积(32, 64)'),
            ('dw_64_128', '深度卷积(64, 128, s=2)'),
            ('dw_128_128', '深度卷积(128, 128)'),
            ('dw_128_256', '深度卷积(128, 256, s=2)'),
            ('dw_256_256', '深度卷积(256, 256)'),
            ('dw_256_512_c4_2', '深度卷积(256, 512)'),
            ('dw_512_512_dil', '深度卷积(512, 512, dil=2)'),
            ('dw_512_512_1', '深度卷积(512, 512)'),
            ('dw_512_512_2', '深度卷积(512, 512)'),
            ('dw_512_512_3', '深度卷积(512, 512)'),
            ('dw_512_512_c5_5', f'深度卷积(512, 512) \n输出: 主干特征 (512, H/8, W/8)')
        ]
        current_node = 'Input'
        for i, (name, label) in enumerate(bb_nodes):
            bb.node(name, label)
            bb.edge(current_node, name)
            current_node = name
        backbone_output_node = current_node

    # CPM 模块
    with dot.subgraph(name='cluster_cpm') as cpm_graph:
        cpm_graph.attr(label='CPM 模块', style='filled', color='lightcoral', fontname=font_name)
        cpm_graph.node('cpm_align', f'卷积(512, {num_channels}, k=1)')
        cpm_graph.node('cpm_trunk', f'序列模块(\n  无BN深度卷积({num_channels}, {num_channels}) (x3)\n)')
        cpm_graph.node('cpm_conv_sum', f'卷积({num_channels}, {num_channels}) + 残差连接')
        cpm_graph.node('cpm_output', f'CPM 特征 ({num_channels}, H/8, W/8)')

        cpm_graph.edge(backbone_output_node, 'cpm_align', ltail='cluster_backbone')
        cpm_graph.edge('cpm_align', 'cpm_trunk')
        cpm_graph.edge('cpm_align', 'cpm_conv_sum', style='dashed', label='残差')
        cpm_graph.edge('cpm_trunk', 'cpm_conv_sum')
        cpm_graph.edge('cpm_conv_sum', 'cpm_output')
    cpm_features_node = 'cpm_output'

    # 初始阶段
    with dot.subgraph(name='cluster_initial_stage') as is_graph:
        is_graph.attr(label='初始预测阶段', style='filled', color='lightgreen', fontname=font_name)
        is_graph.node('is_trunk', f'序列模块(卷积({num_channels}, {num_channels}) (x3))')
        is_graph.node('is_heatmaps_path', f'卷积({num_channels}, 512, k=1)\n卷积(512, {num_heatmaps}, k=1, 无ReLU)')
        is_graph.node('is_pafs_path', f'卷积({num_channels}, 512, k=1)\n卷积(512, {num_pafs}, k=1, 无ReLU)')
        is_graph.node('is_heatmaps_out', f'热力图_0 ({num_heatmaps}, H/8, W/8)', shape='ellipse', color='palegreen')
        is_graph.node('is_pafs_out', f'PAFs_0 ({num_pafs}, H/8, W/8)', shape='ellipse', color='palegreen')

        is_graph.edge(cpm_features_node, 'is_trunk', ltail='cluster_cpm')
        is_graph.edge('is_trunk', 'is_heatmaps_path')
        is_graph.edge('is_trunk', 'is_pafs_path')
        is_graph.edge('is_heatmaps_path', 'is_heatmaps_out')
        is_graph.edge('is_pafs_path', 'is_pafs_out')

    prev_heatmaps_node = 'is_heatmaps_out'
    prev_pafs_node = 'is_pafs_out'

    # 精炼阶段
    for i in range(num_refinement_stages):
        stage_num = i + 1
        with dot.subgraph(name=f'cluster_refinement_stage_{stage_num}') as rs_graph:
            rs_graph.attr(label=f'精炼阶段 {stage_num}', style='filled', color='lightyellow', fontname=font_name)
            concat_in_channels = num_channels + num_heatmaps + num_pafs
            
            rs_graph.node(f'rs_{stage_num}_concat', f'特征拼接\n({concat_in_channels}, H/8, W/8)', shape='invtrapezium', color='khaki')
            
            rs_graph.node(f'rs_{stage_num}_trunk_blocks', f'序列模块(RefinementStageBlock (x5))\n输入: {concat_in_channels}, 输出: {num_channels}')

            rs_graph.node(f'rs_{stage_num}_heatmaps_path', f'卷积({num_channels}, {num_channels}, k=1)\n卷积({num_channels}, {num_heatmaps}, k=1, 无ReLU)')
            rs_graph.node(f'rs_{stage_num}_pafs_path', f'卷积({num_channels}, {num_channels}, k=1)\n卷积({num_channels}, {num_pafs}, k=1, 无ReLU)')
            
            rs_graph.node(f'rs_{stage_num}_heatmaps_out', f'热力图_{stage_num} ({num_heatmaps}, H/8, W/8)', shape='ellipse', color='lemonchiffon')
            rs_graph.node(f'rs_{stage_num}_pafs_out', f'PAFs_{stage_num} ({num_pafs}, H/8, W/8)', shape='ellipse', color='lemonchiffon')

            rs_graph.edge(cpm_features_node, f'rs_{stage_num}_concat', label='CPM特征', style='dashed', constraint='false') 
            rs_graph.edge(prev_heatmaps_node, f'rs_{stage_num}_concat', label=f'来自前阶段热力图', style='dashed')
            rs_graph.edge(prev_pafs_node, f'rs_{stage_num}_concat', label=f'来自前阶段PAFs', style='dashed')
            
            rs_graph.edge(f'rs_{stage_num}_concat', f'rs_{stage_num}_trunk_blocks')
            rs_graph.edge(f'rs_{stage_num}_trunk_blocks', f'rs_{stage_num}_heatmaps_path')
            rs_graph.edge(f'rs_{stage_num}_trunk_blocks', f'rs_{stage_num}_pafs_path')
            rs_graph.edge(f'rs_{stage_num}_heatmaps_path', f'rs_{stage_num}_heatmaps_out')
            rs_graph.edge(f'rs_{stage_num}_pafs_path', f'rs_{stage_num}_pafs_out')

            prev_heatmaps_node = f'rs_{stage_num}_heatmaps_out'
            prev_pafs_node = f'rs_{stage_num}_pafs_out'

    # 最终输出节点
    dot.node('Final_Heatmaps', f'最终热力图 ({num_heatmaps}, H/8, W/8)', color='mediumseagreen', shape='doubleoctagon')
    dot.node('Final_PAFs', f'最终PAFs ({num_pafs}, H/8, W/8)', color='mediumseagreen', shape='doubleoctagon')
    dot.edge(prev_heatmaps_node, 'Final_Heatmaps')
    dot.edge(prev_pafs_node, 'Final_PAFs')
    
    try:
        dot.render(output_filename, view=False, cleanup=True)
        print(f"网络结构图已保存为 {output_filename}.png")
    except Exception as e:
        print(f"渲染图表时出错: {e}")
        print("请确保 Graphviz 已安装并且其路径已添加到系统环境变量 PATH 中。")

if __name__ == '__main__':
    generate_network_diagram()
