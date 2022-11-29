#version 430

in vec2 in_poss;
in vec2 in_vels;
in float in_masses;
in vec2 comp_pos;
in vec2 comp_vel;
in float comp_mass;

out vec2 out_poss;
out vec2 out_vels;
out float out_masses;

void main() {
    float G_CONST = .000006;
    vec2 ut_vels = (comp_pos+comp_vel+comp_mass);
    float d = distance(in_poss, comp_pos);
    if (d > 0)
    {
        d=d*d;
        ut_vels = (G_CONST * (comp_pos - in_poss) * (in_masses + comp_mass)) / d;
    }
    else
    {
        ut_vels = vec2(0,0);
    }
    out_poss = in_poss;
    out_vels = in_vels + ut_vels;
    out_masses = in_masses;
}
