sumo-gui -n network/net/net_cars.net.xml -r simulation/routes.rou.xml --begin 0 --end 600

sumo-gui -n network/net/RandomGrid10_500_tls.net.xml -r simulation/grid_tls_routes.rou.xml --begin 0 --end 3600

netgenerate --grid --grid.number 5 --grid.length 200 \
  --default-junction-type traffic_light \
  -o network/net/RandomGrid5_200_tls.net.xml

python $SUMO_HOME/tools/randomTrips.py \
  -n network/net/RandomGrid5_200_tls.net.xml \
  -r simulation/5_200grid_tls_routes.rou.xml \
  -o simulation/5_200grid_tls_trips.trips.xml \
  -b 0 -e 7200 -p 0.5 --validate

sumo-gui -n network/net/RandomGrid5_200_tls.net.xml -r simulation/5_200grid_tls_routes.rou.xml --begin 0 --end 3600
