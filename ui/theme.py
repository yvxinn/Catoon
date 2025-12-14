"""
UI Theme - CSS 和 Gradio 主题定义

将样式从主文件分离，保持代码整洁。
"""

import gradio as gr


def create_theme() -> gr.themes.Base:
    """创建 Gradio 主题"""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_md,
    )


# 自定义 CSS
CUSTOM_CSS = """
/* 全局字体 */
.gradio-container {
    font-family: 'Helvetica Neue', 'Segoe UI', Roboto, sans-serif;
}

/* 生成按钮样式 */
.generate-btn {
    background: linear-gradient(90deg, #6366f1 0%, #4338ca 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.1s;
}

.generate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
}

/* 遮罩按钮样式 */
.mask-btn {
    font-size: 0.8rem !important;
    padding: 4px 8px !important;
}

/* 核心修改：内部滚动容器 */
.scroll-container {
    max-height: 650px;       /* 限制最大高度 */
    overflow-y: auto;        /* 允许垂直滚动 */
    padding-right: 12px;     /* 给滚动条留出空间 */
    border-radius: 8px;
    background-color: rgba(249, 250, 251, 0.5); /* 极淡的背景色区分 */
    display: block !important; /* 【关键】强制块级布局，防止 Gradio 的 flex 压缩子元素 */
}

/* 手动补充子元素间距 (因为 block 布局不支持 gap) */
.scroll-container > * {
    margin-bottom: 16px;
}

.scroll-container > *:last-child {
    margin-bottom: 0;
}

/* 美化滚动条 */
.scroll-container::-webkit-scrollbar {
    width: 6px;
}

.scroll-container::-webkit-scrollbar-thumb {
    background-color: #cbd5e1;
    border-radius: 4px;
}

.scroll-container::-webkit-scrollbar-track {
    background-color: transparent;
}

/* 标题样式 */
.header h1 {
    margin-bottom: 0.25rem;
}

.header h3 {
    color: #64748b;
    font-weight: normal;
}

/* Tab 样式优化 */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0;
}

.tabs > .tab-nav > button.selected {
    border-bottom: 2px solid #6366f1;
    color: #6366f1;
}

/* 输出图像容器 */
#output_img {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

/* 分组样式 - 内部白色块圆角 */
.gr-group {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    background: white;
}

/* 内部面板/卡片圆角 */
.gr-box, .gr-panel, .gr-form {
    border-radius: 8px !important;
}

/* 输入组件圆角 */
.gr-input, .gr-text-input, .gr-number-input {
    border-radius: 6px !important;
}

.gr-button {
    border-radius: 6px !important;
}

.gr-dropdown, .gr-dropdown > div {
    border-radius: 6px !important;
}

/* 滑块容器圆角 */
.gr-slider {
    border-radius: 6px !important;
}

/* Accordion 样式 */
.gr-accordion {
    border: none !important;
    background: transparent !important;
    border-radius: 8px !important;
}

.gr-accordion > .label-wrap {
    padding: 8px 0;
    border-radius: 6px;
}

/* Accordion 内部面板圆角 */
.gr-accordion > .wrap {
    border-radius: 6px !important;
}

/* 信息提示样式 */
.info-text {
    color: #64748b;
    font-size: 0.85rem;
    font-style: italic;
}
"""


def get_css() -> str:
    """获取自定义 CSS"""
    return CUSTOM_CSS

