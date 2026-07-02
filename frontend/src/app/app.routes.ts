import { Routes } from '@angular/router';
import { Home } from './pages/home/home';
import { Predict } from './pages/predict/predict';
import { ModelInfo } from './pages/model-info/model-info';

export const routes: Routes = [
  { path: '', component: Home },
  { path: 'predict', component: Predict },
  { path: 'model-info', component: ModelInfo },
  { path: '**', redirectTo: '' },
];
