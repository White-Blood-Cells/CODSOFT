package com.example;

public class AdditionOperation {

    public static double add(double num1, double num2) {
        return num1 + num2;
    }

    public static void main(String[] args) {
        double sum = add(5, 10);
        System.out.println("Sum = " + sum);
    }
}
