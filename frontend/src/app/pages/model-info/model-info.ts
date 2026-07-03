import { Component, OnInit } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Api } from '../../services/api';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-model-info',
  imports: [DecimalPipe],
  templateUrl: './model-info.html',
  styleUrl: './model-info.css',
})
export class ModelInfo implements OnInit {
  metrics: any = null;
  recallGapColor = 'green';
  private chart: any = null;

  constructor(private api: Api) {}

  ngOnInit(): void {
    this.api.getMetrics().subscribe({
      next: (res) => {
        this.metrics = res;
        this.recallGapColor = this.getGapColor(res.fairness.recall_gap_male_female);
        setTimeout(() => this.createChart(res), 200);
      },
      error: (err) => console.error(err)
    });
  }

  private getGapColor(gap: number): string {
    if (gap < 0.10) return 'green';
    if (gap <= 0.20) return 'yellow';
    return 'red';
  }

  private createChart(data: any): void {
    const canvas = document.getElementById('fairnessChart') as HTMLCanvasElement;
    if (!canvas) return;
    if (this.chart) this.chart.destroy();
    this.chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1'],
        datasets: [
          {
            label: 'Male',
            data: [
              data.by_gender.male.accuracy,
              data.by_gender.male.precision,
              data.by_gender.male.recall,
              data.by_gender.male.f1
            ],
            backgroundColor: '#4a9eff'
          },
          {
            label: 'Female',
            data: [
              data.by_gender.female.accuracy,
              data.by_gender.female.precision,
              data.by_gender.female.recall,
              data.by_gender.female.f1
            ],
            backgroundColor: '#e94560'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#ffffff' } }
        },
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { color: '#a0a0b0' },
            grid: { color: '#2a2a4a' }
          },
          x: {
            ticks: { color: '#a0a0b0' },
            grid: { color: '#2a2a4a' }
          }
        }
      }
    });
  }
}
