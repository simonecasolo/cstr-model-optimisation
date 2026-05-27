package ProcessSim

model BaseModel
  import Modelica.Constants.eps;
  import Modelica.Constants.R;

  // reactor constants
  constant Real V(quantity="Volume", unit="L") = 150 "Volume of the reactor";
  constant Real Vc(quantity="Volume", unit="L") = 10 "Volume of the cooling jacket";
  constant Real Q(quantity="Volumetric flow rate", unit="L/min") = 100 "Inlet flow rate";

  // reaction constants
  constant Real Ea(quantity="Energy per mole", unit="J/mol") = 83140 "Activation energy";
  constant Real Hr(quantity="Energy per mole", unit="cal/mol") = -2E5 "Heat of reaction";

  // fluid constants
  constant Real Cp(quantity="Heat capacity", unit="cal/g/k") = 1 "Reactant solution heat capacity";
  constant Real Cpc(quantity="Heat capacity", unit="cal/g/k") = 1 "Cooling fluid heat capacity";
  constant Real rho(quantity="Density", unit="g/L") = 1000 "Reactant solution density";
  constant Real rhoc(quantity="Density", unit="g/L") = 1000 "Cooling fluid density";

  // parameters
  parameter Real UA(unit="cal/min/K") = 7E5 "Heat transfer coefficient";
  parameter Real k0(unit="1/min") = 7.2E10 "Pre-exponential factor to k";

  // inlet conditions
  input Real Ci(quantity="Molar concentration", unit="mol/L", start = 0.97, min = 0) "Inlet concentration of the reactant";
  input Real Ti(quantity="Temperature", unit="K", start = 351.5, min = 0) "Inlet temperature of the reactant solution";
  input Real Tci(quantity="Temperature", unit="K", start = 351.6, min = 0) "Inlet temperature of the cooling fluid";
  input Real Qc(quantity="Volumetric flow rate", unit="L/min", start = 150, min = 0) "Flow rate of the cooling fluid";

  // states (outlet conditions)
  output Real C(quantity="Molar concentration", unit="mol/L", min = 0) "Outlet concentration of the reactant";
  output Real T(quantity="Temperature", unit="K", min = 0) "Outlet temperature of the reactant solution";
  output Real Tc(quantity="Temperature", unit="K", min = 0) "Outlet temperature of the cooling fluid";

  // intermediate
  Real k;

equation
// intermediate
  k = k0 * exp(-Ea / (R * T + eps));

// differential equations
  der(C) = Q / V * (Ci - C) - k * C;
  der(T) = Q / V * (Ti - T) - Hr * k * C / (rho * Cp) - UA * (T - Tci) / (rho * Cp * V);
  der(Tc) = Qc / Vc * (Tci - Tc) + UA * (T - Tc) / (rhoc * Cpc * Vc);
end BaseModel;

model SteadyState
  extends BaseModel;

initial equation
  der(C) = 0;
  der(T) = 0;
  der(Tc) = 0;

end SteadyState;

model InitialCondition
  extends BaseModel;

initial equation
  C = Ci;
  T = Ti;
  Tc = Tci;

end InitialCondition;

block CSTR
  import Modelica.Blocks.Interfaces.RealInput;
  import Modelica.Blocks.Interfaces.RealOutput;
  import Modelica.Constants.eps;
  import Modelica.Constants.R;

  // reactor constants
  constant Real V(quantity="Volume", unit="L") = 150 "Volume of the reactor";
  constant Real Vc(quantity="Volume", unit="L") = 10 "Volume of the cooling jacket";
  constant Real Q(quantity="Volumetric flow rate", unit="L/min") = 100 "Inlet flow rate";

  // reaction constants
  constant Real Ea(quantity="Energy per mole", unit="J/mol") = 83140 "Activation energy";
  constant Real Hr(quantity="Energy per mole", unit="cal/mol") = -2E5 "Heat of reaction";

  // fluid constants
  constant Real Cp(quantity="Heat capacity", unit="cal/g/k") = 1 "Reactant solution heat capacity";
  constant Real Cpc(quantity="Heat capacity", unit="cal/g/k") = 1 "Cooling fluid heat capacity";
  constant Real rho(quantity="Density", unit="g/L") = 1000 "Reactant solution density";
  constant Real rhoc(quantity="Density", unit="g/L") = 1000 "Cooling fluid density";

  // parameters
  parameter Real UA(unit="cal/min/K") = 7E5 "Heat transfer coefficient";
  parameter Real k0(unit="1/min") = 7.2E10 "Pre-exponential factor to k";

  // inlet conditions
  RealInput Ci(quantity="Molar concentration", unit="mol/L", min = 0) "Inlet concentration of the reactant" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-55, 29}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  RealInput Ti(quantity="Temperature", unit="K", min = 0) "Inlet temperature of the reactant" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-55, 9}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  RealInput Qc(quantity="Volumetric flow rate", unit="L/min", min = 0) "Flow rate of the cooling fluid" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-55, -29}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  RealInput Tci(quantity="Temperature", unit="K", min = 0) "Inlet temperature of the cooling fluid" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-55, -49}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));

  // states (outlet conditions)
  RealOutput T(quantity="Temperature", unit="K", min = 0) "Outlet temperature of the reactant" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {60, 20}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  RealOutput C(quantity="Molar concentration", unit="mol/L", min = 0) "Outlet concentration of the reactant" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {60, -6}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  RealOutput Tc(quantity="Temperature", unit="K", min = 0) "Outlet temperature of the cooling fluid" annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {60, -34}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));

  // intermediate
  Real k;

initial equation
  der(C) = 0;
  der(T) = 0;
  der(Tc) = 0;

equation
  // intermediate
  k = k0 * exp(-Ea / (R * T + eps));

  // differential equations
  der(C) = Q / V * (Ci - C) - k * C;
  der(T) = Q / V * (Ti - T) - Hr * k * C / (rho * Cp) - UA * (T - Tci) / (rho * Cp * V);
  der(Tc) = Qc / Vc * (Tci - Tc) + UA * (T - Tc) / (rhoc * Cpc * Vc);
  annotation(
    Icon(graphics = {Polygon(origin = {-8, 4}, lineColor = {154, 153, 150}, fillColor = {246, 245, 244}, fillPattern = FillPattern.VerticalCylinder, lineThickness = 1, points = {{-40, 40}, {-40, -60}, {-20, -80}, {40, -80}, {60, -60}, {60, 40}, {40, 60}, {-20, 60}, {-20, 60}, {-40, 40}}), Text(origin = {-3, 78}, extent = {{-33, 28}, {33, -28}}, textString = "%name"), Ellipse(origin = {15, -45}, fillColor = {119, 118, 123}, fillPattern = FillPattern.Sphere, extent = {{-15, 5}, {15, -5}}), Ellipse(origin = {-13, -45}, fillColor = {119, 118, 123}, pattern = LinePattern.None, fillPattern = FillPattern.Sphere, lineThickness = 0, extent = {{-15, 5}, {15, -5}}), Rectangle(origin = {1, 9}, fillColor = {119, 118, 123}, pattern = LinePattern.None, fillPattern = FillPattern.VerticalCylinder, lineThickness = 0, extent = {{-1, 55}, {1, -55}})}));

end CSTR;

model TestCSTR
  ProcessSim.CSTR cstr annotation(
    Placement(visible = true, transformation(origin = {48, 14}, extent = {{-24, -24}, {24, 24}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant Ti(k = 352) annotation(
    Placement(visible = true, transformation(origin = {-62, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant Qc(k = 150) annotation(
    Placement(visible = true, transformation(origin = {-62, -22}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant Tci(k = 345) annotation(
    Placement(visible = true, transformation(origin = {-62, -64}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Step Ci(height = 1.5, offset = 0.97, startTime = 4)  annotation(
      Placement(visible = true, transformation(origin = {-62, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));

  equation
    connect(Ti.y, cstr.Ti) annotation(
        Line(points = {{-51, 16}, {33, 16}}, color = {0, 0, 127}));
    connect(Qc.y, cstr.Qc) annotation(
        Line(points = {{-51, -22}, {-27, -22}, {-27, 7}, {33, 7}}, color = {0, 0, 127}));
    connect(Tci.y, cstr.Tci) annotation(
        Line(points = {{-51, -64}, {-3, -64}, {-3, 3}, {33, 3}, {33, 2}}, color = {0, 0, 127}));
    connect(Ci.y, cstr.Ci) annotation(
        Line(points = {{-51, 58}, {-11, 58}, {-11, 21}, {33, 21}, {33, 20}}, color = {0, 0, 127}));

end TestCSTR;

annotation(
    uses(Modelica(version = "4.0.0")));

end ProcessSim;