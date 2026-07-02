import { Component, OnInit } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Api } from '../../services/api';

@Component({
  selector: 'app-model-info',
  imports: [DecimalPipe],
  templateUrl: './model-info.html',
  styleUrl: './model-info.css',
})
export class ModelInfo implements OnInit{
  metrics: any = null;
  constructor(private api: Api) {}
  ngOnInit(): void {
  this.api.getMetrics().subscribe({
    next: (res) => this.metrics = res,
    error: (err) => console.error(err)
  });
}

}
