
### macro f_code(obj)
    
    ia_type qr = x;
    ia_type qi = y;
    ia_type qj = z;
    ia_type qk = ia_new(0,0);
    
    const float cr = {{ obj.c_code[0] }},
        ci = {{ obj.c_code[1] }},
        cj = {{ obj.c_code[2] }},
        ck = {{ obj.c_code[3] }};
    
    ia_type qr1;
    
    for (int iii=0; iii<{{ obj.julia_itr }}; ++iii)
    {
        // Quaternion operation z -> z^2 + c
        // lazy... "should" use ia_add
        qr1 = ia_sub(ia_pow2(qr), ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk))+cr;
        qi = 2 * ia_mul(qr,qi) + ci;
        qj = 2 * ia_mul(qr,qj) + cj;
        qk = 2 * ia_mul(qr,qk) + ck;
        qr = qr1;
    }
    
    f = ia_pow2(qr)+ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk) - 4.0;
    
### endmacro

### macro df_code(obj)
    df = 1; // TODO
### endmacro
