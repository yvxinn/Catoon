"""
Gradio UI - 交互式卡通化界面 (Professional Version)

重构后的简洁入口文件，业务逻辑和 UI 布局已分离到独立模块。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """主入口"""
    from ui.layout import create_ui
    
    # 创建 UI（事件绑定已在 create_ui 内部完成）
    ui_components = create_ui()
    demo = ui_components["demo"]
    theme = ui_components.get("theme")
    css = ui_components.get("css")
    
    # 启动服务（Gradio 6.0+ 需要在 launch 时传递 theme 和 css）
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
