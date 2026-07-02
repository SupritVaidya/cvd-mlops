import { Component } from '@angular/core';
import { Api } from '../../services/api';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';


@Component({
  selector: 'app-predict',
  imports: [FormsModule, DecimalPipe],
  templateUrl: './predict.html',
  styleUrl: './predict.css',
})
export class Predict {

    form = {
    Age: 0,
    Sex: '',
    ChestPainType: '',
    RestingBP: 0,
    Cholesterol: 0,
    FastingBS: 0,
    RestingECG: '',
    MaxHR: 0,
    ExerciseAngina: '',
    Oldpeak: 0,
    ST_Slope: ''
  };

  result: any = null;


  constructor(private api: Api) {}

  submit(): void {
    this.api.predict(this.form).subscribe({
      next: (res) => this.result = res,
      error: (err) => console.error(err)
    });
  }

}

