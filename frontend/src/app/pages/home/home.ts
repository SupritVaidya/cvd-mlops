import { Component, OnInit } from '@angular/core';
import { Api } from '../../services/api';


@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.html',
  styleUrl: './home.css',
})
export class Home implements OnInit{

  healthStatus = '';
  constructor(private api: Api) {}
  ngOnInit(): void {
  this.api.getHealth().subscribe({
    next: (res) => this.healthStatus = 'Backend is up',
    error: () => this.healthStatus = 'Backend is down'
  });
}



}
