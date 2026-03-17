package com.example;

import org.junit.Test;
import static org.junit.Assert.*;

public class AdditionOperationTest {

    @Test
    public void testAddition() {
        double result = AdditionOperation.add(5, 10);
        assertEquals(15.0, result, 0.01);
    }
}
