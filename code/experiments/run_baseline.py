import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版：主实验E1 - Baseline (无路由的纯微调模型)
统一使用LLM增强评估（与E2、E3评分标准一致）
功能：测试单一微调模型在所有任务上的性能
日期：2026-05-24
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

# 添加项目路径
sys.path.insert(0, '/data/nefu/毕业论文实验')

# ✅ 修改：导入LLM增强评估模块（与E3保持一致）
from metrics_llm_enhanced import LLMEnhancedMetricsCalculator

# 配置日志
log_dir = "/data/nefu/毕业论文实验/experiments/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'E1_baseline_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BaselineExperiment:
    """E1 Baseline实验主类"""
    
    def __init__(self):
        # 路径配置
        self.base_model_path = "/data/nefu/homee/Qwen/Qwen1.5-7B-Chat"
        self.lora_path = "/data/nefu/毕业论文实验/checkpoints/Qwen1.5-7B-Chat_lora/checkpoint-74170"
        self.test_file = "/data/nefu/毕业论文实验/experiments/results/test_all_tasks.jsonl"
        
        # DeepSeek API配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 输出配置
        self.output_dir = "/data/nefu/毕业论文实验/experiments/results/main"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_file = os.path.join(self.output_dir, "E1_baseline_fixed_results.json")
        self.detailed_file = os.path.join(self.output_dir, "E1_baseline_fixed_detailed.jsonl")
        
        # 模型和评估器
        self.model = None
        
        # ✅ 修改：使用LLM增强评估器（与E3一致）
        self.metrics_calculator = LLMEnhancedMetricsCalculator(
            deepseek_api_key=self.deepseek_api_key,
            qwen_model=None  # 稍后在load_model中赋值
        )
        
        # ✅ 修改：统计数据结构与E3保持一致
        self.results = []
        self.task_stats = defaultdict(lambda: {
            'total': 0,
            'answer_correct': 0,
            'scores': [],
            'response_lengths': [],
            'processing_times': [],
            # Biomass详细维度
            'species_scores': [],
            'region_scores': [],
            'component_scores': [],
            'variable_scores': [],
            'form_scores': [],
            'coefficient_scores': [],
            'fallback_region_count': 0,
            'fallback_species_count': 0,
            # Forestry/General详细维度
            'accuracy_scores': [],
            'completeness_scores': [],
            'relevance_scores': []
        })
        
        self.error_count = 0
        self.empty_response_count = 0
    
    def load_model(self):
        """加载微调模型"""
        logger.info("=" * 80)
        logger.info("🔧 加载模型...")
        logger.info(f"  基础模型: {self.base_model_path}")
        logger.info(f"  LoRA模型: {self.lora_path}")
        
        try:
            from modules.qwen_model_adapter import QwenModelAdapter
            
            self.model = QwenModelAdapter(
                base_model_path=self.base_model_path,
                lora_path=self.lora_path
            )
            
            self.model.load_model()
            logger.info("✅ 模型加载成功")
            
            # ✅ 将Qwen模型传递给评估器（用于LLM兜底）
            self.metrics_calculator.set_qwen_model(self.model)
            logger.info("✅ 评估器配置完成（支持LLM兜底）")
            
        except ImportError as e:
            logger.error(f"❌ 导入模块失败: {e}")
            logger.error("请确保 modules.qwen_model_adapter 存在")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            sys.exit(1)
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        logger.info("=" * 80)
        logger.info(f"📂 加载测试数据: {self.test_file}")
        
        if not os.path.exists(self.test_file):
            logger.error(f"❌ 测试文件不存在: {self.test_file}")
            logger.info("💡 请先运行 prepare_unified_test_sets.py 生成测试集")
            sys.exit(1)
        
        test_data = []
        
        try:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        test_data.append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"行 {idx + 1} JSON解析失败，跳过")
                        continue
            
            logger.info(f"✅ 成功加载 {len(test_data)} 条测试样本")
            
            # 统计各任务数量
            task_counts = defaultdict(int)
            for data in test_data:
                task_type = data.get('task_type', 'unknown')
                task_counts[task_type] += 1
            
            logger.info("📊 任务分布:")
            for task, count in sorted(task_counts.items()):
                logger.info(f"  - {task}: {count} ({count/len(test_data)*100:.1f}%)")
            
            return test_data
            
        except Exception as e:
            logger.error(f"❌ 加载测试数据失败: {e}", exc_info=True)
            sys.exit(1)
    
    def evaluate_single_sample(self, data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """评估单个样本"""
        task_type = data.get('task_type', 'unknown')
        instruction = data.get('instruction', '').strip()
        
        # 获取参考答案（生物量用方程，其他用output）
        if task_type == 'biomass':
            reference = data.get('equation', '')
        else:
            reference = data.get('output', '')
        
        if not instruction.strip():
            logger.warning(f"样本 {idx} instruction为空，跳过")
            return None
        
        # 生成预测
        start_time = time.time()
        predicted_answer = ""
        error_message = None
        
        try:
            predicted_answer = self.model.generate(instruction)
            
            if not predicted_answer:
                self.empty_response_count += 1
                logger.warning(f"样本 {idx} 模型返回空响应")
                
        except Exception as e:
            self.error_count += 1
            error_message = str(e)
            logger.error(f"样本 {idx} 生成失败: {e}")
        
        processing_time = time.time() - start_time
        
        # ✅ 修改：使用LLM增强评估（与E3一致）
        if task_type == 'biomass':
            # Biomass：多维度评分 + LLM兜底
            metrics = self.metrics_calculator.calculate_biomass_metrics(
                predicted_answer=predicted_answer,
                reference=reference,
                instruction=instruction,
                equation_db=None  # E1没有方程库
            )
        else:
            # Forestry/General：准确性、完整性、相关性
            metrics = self.metrics_calculator.calculate_qa_metrics(
                predicted=predicted_answer,
                reference=reference,
                question=instruction,
                task_type=task_type
            )
        
        # ✅ 修改：更新统计（与E3一致）
        self.task_stats[task_type]['total'] += 1
        self.task_stats[task_type]['answer_correct'] += metrics['score']
        self.task_stats[task_type]['scores'].append(metrics['score'])
        self.task_stats[task_type]['response_lengths'].append(len(predicted_answer))
        self.task_stats[task_type]['processing_times'].append(processing_time)
        
        # 记录详细维度
        if task_type == 'biomass':
            # Biomass多维度得分
            details = metrics.get('details', {})
            self.task_stats[task_type]['species_scores'].append(details.get('species', 0))
            self.task_stats[task_type]['region_scores'].append(details.get('region', 0))
            self.task_stats[task_type]['component_scores'].append(details.get('component', 0))
            self.task_stats[task_type]['variable_scores'].append(details.get('variables', 0))
            self.task_stats[task_type]['form_scores'].append(details.get('form', 0))
            self.task_stats[task_type]['coefficient_scores'].append(details.get('coefficient', 0))
            
            # 兜底策略统计
            if details.get('used_fallback_region'):
                self.task_stats[task_type]['fallback_region_count'] += 1
            if details.get('used_fallback_species'):
                self.task_stats[task_type]['fallback_species_count'] += 1
                
        elif task_type in ['forestry_qa', 'general_chat']:
            # Forestry/General三维度得分
            self.task_stats[task_type]['accuracy_scores'].append(metrics.get('accuracy', 0))
            self.task_stats[task_type]['completeness_scores'].append(metrics.get('completeness', 0))
            self.task_stats[task_type]['relevance_scores'].append(metrics.get('relevance', 0))
        
        # 构建结果
        result = {
            'index': idx,
            'task_type': task_type,
            'true_task_type': task_type,  # 与E2/E3保持一致
            'instruction': instruction,
            'reference': reference,
            'predicted_answer': predicted_answer,
            'metrics': metrics,
            'score': metrics['score'],
            'processing_time': processing_time,
            'response_length': len(predicted_answer),
            'error': error_message,
            'metadata': data.get('metadata', {})
        }
        
        return result
    
    def run_evaluation(self):
        """运行完整评估"""
        logger.info("=" * 80)
        logger.info("🚀 开始E1实验评估...")
        logger.info(f"  实验: Baseline (无路由的纯微调模型)")
        logger.info(f"  评估方法: LLM增强评估（与E2/E3一致）")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载测试数据
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        # 打开详细结果文件
        detailed_file_handle = open(self.detailed_file, 'w', encoding='utf-8')
        
        start_time = time.time()
        
        # 逐样本评估
        for idx, data in enumerate(test_data):
            result = self.evaluate_single_sample(data, idx)
            
            if result:
                self.results.append(result)
                detailed_file_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 定期输出进度
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (idx + 1) / elapsed
                eta_seconds = (total_samples - idx - 1) / samples_per_sec if samples_per_sec > 0 else 0
                
                # 计算当前整体准确率
                total_evaluated = sum(s['total'] for s in self.task_stats.values())
                total_correct = sum(s['answer_correct'] for s in self.task_stats.values())
                current_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
                
                logger.info(
                    f"  进度: {idx + 1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%) | "
                    f"平均得分: {current_accuracy:.2%} | "
                    f"速度: {samples_per_sec:.2f} 样本/秒 | "
                    f"预计剩余: {eta_seconds/60:.1f} 分钟"
                )
        
        detailed_file_handle.close()
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"✅ 评估完成! 总耗时: {total_time/60:.1f} 分钟")
        logger.info(f"  处理样本数: {len(self.results)}/{total_samples}")
        logger.info(f"  错误数: {self.error_count}")
        logger.info(f"  空响应数: {self.empty_response_count}")
    
    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """计算汇总指标"""
        logger.info("=" * 80)
        logger.info("📊 计算汇总指标...")
        
        summary = {
            'experiment_name': 'E1-Baseline-Fixed',
            'model_type': 'QwenLoRA_NoRouter',
            'evaluation_method': 'LLM_Enhanced_Unified',
            'total_samples': len(self.results),
            'error_count': self.error_count,
            'empty_response_count': self.empty_response_count,
            'task_performance': {}
        }
        
        # 计算每个任务的详细指标
        for task_type, stats in self.task_stats.items():
            if stats['total'] == 0:
                continue
            
            task_metrics = {
                'total_samples': stats['total'],
                'answer_accuracy': stats['answer_correct'] / stats['total'],
                'average_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                'std_score': np.std(stats['scores']) if stats['scores'] else 0.0,
                'min_score': np.min(stats['scores']) if stats['scores'] else 0.0,
                'max_score': np.max(stats['scores']) if stats['scores'] else 0.0,
                'average_response_length': np.mean(stats['response_lengths']) if stats['response_lengths'] else 0.0,
                'average_processing_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0.0
            }
            
            # 添加详细维度统计
            if task_type == 'biomass':
                task_metrics.update({
                    'average_species_match': np.mean(stats['species_scores']) if stats['species_scores'] else 0.0,
                    'average_region_match': np.mean(stats['region_scores']) if stats['region_scores'] else 0.0,
                    'average_component_match': np.mean(stats['component_scores']) if stats['component_scores'] else 0.0,
                    'average_variable_match': np.mean(stats['variable_scores']) if stats['variable_scores'] else 0.0,
                    'average_form_match': np.mean(stats['form_scores']) if stats['form_scores'] else 0.0,
                    'average_coefficient_similarity': np.mean(stats['coefficient_scores']) if stats['coefficient_scores'] else 0.0,
                    'fallback_region_count': stats['fallback_region_count'],
                    'fallback_species_count': stats['fallback_species_count'],
                    'fallback_region_rate': stats['fallback_region_count'] / stats['total'] if stats['total'] > 0 else 0.0,
                    'fallback_species_rate': stats['fallback_species_count'] / stats['total'] if stats['total'] > 0 else 0.0
                })
                
                logger.info(f"\n📈 {task_type} 性能:")
                logger.info(f"  - 样本数: {task_metrics['total_samples']}")
                logger.info(f"  - 答案准确率: {task_metrics['answer_accuracy']:.2%}")
                logger.info(f"  - 平均得分: {task_metrics['average_score']:.2f} ± {task_metrics['std_score']:.2f}")
                logger.info(f"  📊 多维度匹配:")
                logger.info(f"    - 树种: {task_metrics['average_species_match']:.2f}")
                logger.info(f"    - 地区: {task_metrics['average_region_match']:.2f}")
                logger.info(f"    - 组分: {task_metrics['average_component_match']:.2f}")
                logger.info(f"    - 参数变量: {task_metrics['average_variable_match']:.2f}")
                logger.info(f"    - 方程形式: {task_metrics['average_form_match']:.2f}")
                logger.info(f"    - 系数相似度: {task_metrics['average_coefficient_similarity']:.2f}")
                logger.info(f"  🔧 LLM兜底:")
                logger.info(f"    - 地区兜底: {stats['fallback_region_count']} 次 ({task_metrics['fallback_region_rate']:.1%})")
                logger.info(f"    - 树种兜底: {stats['fallback_species_count']} 次 ({task_metrics['fallback_species_rate']:.1%})")
                
            elif task_type in ['forestry_qa', 'general_chat']:
                task_metrics.update({
                    'average_accuracy': np.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0.0,
                    'average_completeness': np.mean(stats['completeness_scores']) if stats['completeness_scores'] else 0.0,
                    'average_relevance': np.mean(stats['relevance_scores']) if stats['relevance_scores'] else 0.0
                })
                
                logger.info(f"\n📈 {task_type} 性能:")
                logger.info(f"  - 样本数: {task_metrics['total_samples']}")
                logger.info(f"  - 答案质量: {task_metrics['answer_accuracy']:.2%}")
                logger.info(f"  - 平均得分: {task_metrics['average_score']:.2f} ± {task_metrics['std_score']:.2f}")
                logger.info(f"  📊 详细维度:")
                logger.info(f"    - 准确性: {task_metrics['average_accuracy']:.1f}/10")
                logger.info(f"    - 完整性: {task_metrics['average_completeness']:.1f}/10")
                logger.info(f"    - 相关性: {task_metrics['average_relevance']:.1f}/10")
            
            logger.info(f"  - 平均响应长度: {task_metrics['average_response_length']:.1f} 字符")
            logger.info(f"  - 平均处理时间: {task_metrics['average_processing_time']:.3f} 秒")
            
            summary['task_performance'][task_type] = task_metrics
        
        # 计算整体指标
        total_samples = sum(s['total'] for s in self.task_stats.values())
        
        if total_samples > 0:
            overall_metrics = {
                'answer_accuracy': sum(s['answer_correct'] for s in self.task_stats.values()) / total_samples,
                'average_score': np.mean([score for s in self.task_stats.values() for score in s['scores']]),
                'average_processing_time': np.mean([t for s in self.task_stats.values() for t in s['processing_times']])
            }
            
            summary['overall_performance'] = overall_metrics
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 整体性能:")
            logger.info(f"  - 总样本数: {total_samples}")
            logger.info(f"  - 整体准确率: {overall_metrics['answer_accuracy']:.2%}")
            logger.info(f"  - 平均得分: {overall_metrics['average_score']:.2f}")
            logger.info(f"  - 平均处理时间: {overall_metrics['average_processing_time']:.3f} 秒/样本")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """保存结果"""
        logger.info("=" * 80)
        logger.info(f"💾 保存结果到: {self.output_file}")
        
        full_results = {
            'metadata': {
                'experiment': 'E1-Baseline-Fixed',
                'timestamp': datetime.now().isoformat(),
                'base_model': self.base_model_path,
                'lora_model': self.lora_path,
                'test_file': self.test_file,
                'evaluation_method': 'LLM_Enhanced_Unified'
            },
            'summary': summary,
            'sample_results': self.results[:200]
        }
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 汇总结果已保存: {self.output_file}")
            logger.info(f"✅ 详细结果已保存: {self.detailed_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}", exc_info=True)
    
    def run(self):
        """运行完整的E1实验"""
        logger.info("🎯 开始执行 E1-Baseline-Fixed 实验")
        logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 运行评估
        self.run_evaluation()
        
        # 3. 计算汇总指标
        summary = self.calculate_summary_metrics()
        
        # 4. 保存结果
        self.save_results(summary)
        
        logger.info("=" * 80)
        logger.info("🎉 E1-Baseline-Fixed 实验完成!")
        logger.info(f"下一步: 运行 E2-Fixed-Rule-Routing 实验进行对比")
        logger.info("=" * 80)


if __name__ == "__main__":
    experiment = BaselineExperiment()
    experiment.run()