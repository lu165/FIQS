import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控脚本：FIQS实验完整流程
TaYOUR_DEEPSEEK_API_KEY Adaptive Routing (FIQS) 实验管理器
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json

class FIQSEexperimentManager:
    """FIQS实验管理器"""
    
    def __init__(self, base_dir: str = "/data/nefu/毕业论文实验"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "experiments" / "results"
        self.code_dir = self.base_dir / "code"
        
        self.experiments = {
            'E1': {
                'name': '纯微调Baseline',
                'script': self.code_dir / 'test_E1_baseline_final.py',
                'output': self.results_dir / 'E1_baseline_detailed.jsonl',
                'description': '所有任务都用微调Qwen模型',
                'status': 'pending'
            },
            'E2': {
                'name': '纯RAG系统',
                'script': 'E2_pure_rag_experiment.py',  # 新创建的
                'output': self.results_dir / 'E2_pure_rag_results.jsonl',
                'description': '所有任务都用RAG策略',
                'status': 'pending'
            },
            'E3': {
                'name': '智能路由系统',
                'script': self.code_dir / 'test_E2_router_system_final.py',
                'output': self.results_dir / 'main' / 'E2_router_system_detailed.jsonl',
                'description': '根据任务类型智能选择策略',
                'status': 'running'  # 当前正在运行
            },
            'E4': {
                'name': '随机路由（消融）',
                'script': 'E4_ablation_random_router.py',  # 新创建的
                'output': self.results_dir / 'E4_random_router_results.jsonl',
                'description': '随机选择策略，验证智能路由价值',
                'status': 'pending'
            }
        }
    
    def print_banner(self, text: str):
        """打印横幅"""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80 + "\n")
    
    def check_experiment_status(self):
        """检查各实验状态"""
        self.print_banner("📊 实验状态检查")
        
        for exp_id, exp_info in self.experiments.items():
            output_file = exp_info['output']
            
            if output_file.exists():
                # 统计样本数
                with open(output_file, 'r', encoding='utf-8') as f:
                    sample_count = sum(1 for _ in f)
                
                exp_info['status'] = 'completed' if sample_count >= 3000 else 'partial'
                exp_info['sample_count'] = sample_count
            else:
                exp_info['status'] = 'not_started'
                exp_info['sample_count'] = 0
        
        # 显示状态
        for exp_id, exp_info in self.experiments.items():
            status_emoji = {
                'completed': '✅',
                'running': '🔄',
                'partial': '⚠️',
                'pending': '⏳',
                'not_started': '❌'
            }
            
            emoji = status_emoji.get(exp_info['status'], '❓')
            status_text = exp_info['status'].upper()
            
            print(f"{emoji} {exp_id} - {exp_info['name']}")
            print(f"   状态: {status_text}")
            print(f"   样本: {exp_info.get('sample_count', 0)}/3463")
            print(f"   说明: {exp_info['description']}")
            print()
    
    def run_experiment(self, exp_id: str, background: bool = False):
        """运行单个实验"""
        exp_info = self.experiments[exp_id]
        script = exp_info['script']
        
        self.print_banner(f"🚀 启动实验: {exp_id} - {exp_info['name']}")
        
        print(f"脚本: {script}")
        print(f"输出: {exp_info['output']}")
        
        if not Path(script).exists():
            print(f"❌ 脚本不存在: {script}")
            return False
        
        if background:
            # 后台运行
            log_file = self.results_dir / f"{exp_id}_experiment.log"
            cmd = f"nohup python3 {script} > {log_file} 2>&1 &"
            print(f"\n后台运行命令:")
            print(f"  {cmd}")
            
            subprocess.Popen(cmd, shell=True)
            print(f"\n✅ 实验已在后台启动")
            print(f"📝 日志文件: {log_file}")
            print(f"💡 监控命令: tail -f {log_file}")
        else:
            # 前台运行
            cmd = f"python3 {script}"
            print(f"\n运行命令: {cmd}\n")
            
            try:
                result = subprocess.run(cmd, shell=True, check=True)
                print(f"\n✅ 实验完成")
                return True
            except subprocess.CalledProcessError as e:
                print(f"\n❌ 实验失败: {e}")
                return False
    
    def run_all_missing_experiments(self):
        """运行所有缺失的实验"""
        self.print_banner("🔄 批量运行缺失实验")
        
        # 检查状态
        self.check_experiment_status()
        
        # 找出需要运行的实验
        to_run = []
        for exp_id, exp_info in self.experiments.items():
            if exp_info['status'] in ['not_started', 'pending']:
                to_run.append(exp_id)
        
        if not to_run:
            print("✅ 所有实验都已完成！")
            return
        
        print(f"需要运行的实验: {', '.join(to_run)}")
        print(f"\n确认启动 {len(to_run)} 个实验？(y/n): ", end='')
        
        confirm = input().strip().lower()
        if confirm != 'y':
            print("❌ 取消运行")
            return
        
        # 依次启动
        for exp_id in to_run:
            self.run_experiment(exp_id, background=True)
            time.sleep(5)  # 等待5秒再启动下一个
    
    def analyze_results(self):
        """分析所有实验结果"""
        self.print_banner("📊 分析实验结果")
        
        # 运行分析脚本
        analysis_script = "analyze_all_experiments.py"
        
        if not Path(analysis_script).exists():
            print(f"❌ 分析脚本不存在: {analysis_script}")
            return
        
        cmd = f"python3 {analysis_script}"
        print(f"运行: {cmd}\n")
        
        subprocess.run(cmd, shell=True)
    
    def show_menu(self):
        """显示菜单"""
        while True:
            self.print_banner("🎯 FIQS实验管理系统")
            
            print("请选择操作:")
            print()
            print("1. 📊 检查实验状态")
            print("2. 🚀 运行单个实验")
            print("3. 🔄 运行所有缺失实验")
            print("4. 📈 分析实验结果")
            print("5. 📝 查看实验说明")
            print("6. 🚪 退出")
            print()
            
            choice = input("请输入选项 (1-6): ").strip()
            
            if choice == '1':
                self.check_experiment_status()
                input("\n按Enter继续...")
                
            elif choice == '2':
                print("\n可选实验:")
                for exp_id, exp_info in self.experiments.items():
                    print(f"  {exp_id}: {exp_info['name']}")
                
                exp_id = input("\n输入实验编号 (E1/E2/E3/E4): ").strip().upper()
                
                if exp_id in self.experiments:
                    background = input("后台运行？(y/n): ").strip().lower() == 'y'
                    self.run_experiment(exp_id, background)
                else:
                    print("❌ 无效的实验编号")
                
                input("\n按Enter继续...")
                
            elif choice == '3':
                self.run_all_missing_experiments()
                input("\n按Enter继续...")
                
            elif choice == '4':
                self.analyze_results()
                input("\n按Enter继续...")
                
            elif choice == '5':
                self.show_experiment_description()
                input("\n按Enter继续...")
                
            elif choice == '6':
                print("\n👋 再见！")
                break
                
            else:
                print("❌ 无效选项")
                time.sleep(1)
    
    def show_experiment_description(self):
        """显示实验说明"""
        self.print_banner("📝 FIQS实验方案说明")
        
        print("""
🎯 研究目标：
  验证任务自适应路由系统（FIQS）在林业多任务场景下的有效性

📊 实验设计：

  E1 - 纯微调Baseline
  ├─ 策略: 所有任务都用微调Qwen模型
  ├─ 目的: 作为基线对比
  └─ 预期: 生物量任务效果差 (~0%)

  E2 - 纯RAG系统
  ├─ 策略: 所有任务都用检索增强生成
  ├─ 目的: 验证RAG的普适性
  └─ 预期: 整体效果中等 (~65%)

  E3 - 智能路由系统 ⭐
  ├─ 策略: 根据任务类型智能选择
  │   ├─ 生物量 → 专业模块（规则+知识库）
  │   ├─ 林业问答 → 微调+RAG混合
  │   └─ 通用对话 → 纯微调
  ├─ 目的: 主实验，证明FIQS有效性
  └─ 预期: 各任务都达到最优 (~85%+)

  E4 - 随机路由（消融实验）
  ├─ 策略: 随机选择微调/RAG/混合策略
  ├─ 目的: 验证智能路由的必要性
  └─ 预期: 效果不如E3 (~70%)

📈 评估指标：
  - 整体准确率
  - 分任务类型准确率（生物量/林业问答/通用对话）
  - 路由准确率（E3）

📄 论文贡献：
  1. 提出FIQS框架：任务感知的自适应路由
  2. 在林业多任务场景验证有效性
  3. 整体性能提升15%+
""")


def main():
    """主函数"""
    manager = FIQSEexperimentManager()
    
    # 如果有命令行参数，执行对应命令
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'status':
            manager.check_experiment_status()
        elif cmd == 'run':
            if len(sys.argv) > 2:
                exp_id = sys.argv[2].upper()
                background = '--bg' in sys.argv
                manager.run_experiment(exp_id, background)
            else:
                manager.run_all_missing_experiments()
        elif cmd == 'analyze':
            manager.analyze_results()
        else:
            print(f"❌ 未知命令: {cmd}")
            print("用法: python3 main_experiment_manager.py [status|run|analyze]")
    else:
        # 交互式菜单
        manager.show_menu()


if __name__ == "__main__":
    main()