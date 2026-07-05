import { Component, OnInit } from '@angular/core';
import { Api } from '../../services/api';
import { retry, delay, catchError } from 'rxjs/operators';
import { timer } from 'rxjs';
import { retryWhen, mergeMap } from 'rxjs/operators';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.html',
  styleUrl: './home.css',
})
export class Home implements OnInit {

  healthStatus = '';
  constructor(private api: Api) {}

  ngOnInit(): void {
    this.healthStatus = 'Checking connection...';
    this.api.getHealth().pipe(
      retryWhen(errors =>
        errors.pipe(
          mergeMap((error, index) => {
            if (index >= 5) throw error;
            return timer(8000);
          })
        )
      )
    ).subscribe({
      next: () => this.healthStatus = 'Backend is up',
      error: () => this.healthStatus = 'Backend is down'
    });
  }
}