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

    form: any = {
    Age: null,
    Sex: '',
    ChestPainType: '',
    RestingBP: null,
    Cholesterol: null,
    FastingBS: '',
    RestingECG: '',
    MaxHR: null,
    ExerciseAngina: '',
    Oldpeak: null,
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

