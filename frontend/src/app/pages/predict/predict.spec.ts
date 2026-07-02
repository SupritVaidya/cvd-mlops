import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Predict } from './predict';

describe('Predict', () => {
  let component: Predict;
  let fixture: ComponentFixture<Predict>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Predict]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Predict);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
