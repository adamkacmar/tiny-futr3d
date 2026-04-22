- VoxelNet: ci trenovali block procesovania (Figure 4 nalavo) tymto klasickym sposobom alebo zobrali len architekturu a pouzili ju na ten query-based training
- https://github.com/CuberrChen/VoxelNet/blob/main/README_EN.md
- normalne trenuju 20 epoch v setupe, ktory som ja hovoril, nic explicitne nepisu, ze by to tak nerobili
- co tam nie je napisane: query inicializaciu su na learned embeddingoch
- backbone vies spojit lebo nemaju rozdielne vahy ale query inicializacie su rozne a oni nepisu ci beru jednu alebo druhu, alebo co s nimi robia
- note: VoxelNeXt existuje na nuScenes predtrenovany
hladanie pre-trained vah, ako to oni trenovali a ci existuje nejaky baseline
- zistit ako baselineovo bude nieco take bezat
- stiahnut predtrenovane vahy na architekturu
- kolko to ma trenovatelnych parametrov
- rozdiel medzi metodami PointPillars a FUTR3D - pozriet TOTO

- skus si inicializovat model a vyprintovat pocet trenovatelnych parametrov a ten VoxelNet je asi maly aj PointPillars
- cize mozno to nebudem musiet zreplikovat, len pridat Radar a vyhodnotit vysledky

- urobili overkill zepouzili ResNet101 na RGB obrazky

DP1:
- prist na to v akych najadverznejsich podmienkach vie fungovat stale dobre
- overit, ze ked pustim nejaky trening, tak ze sa to podoba na tie vysledky co oni dosiahli
- potom sa vieme pozriet na modifikaciu ich feature sampleru (MAFS), s nejakou cielenou inicializaciou, pozriet sa na nejake vydhonotenia, baseline, nejaku cielenu inicializaciu
- co mozem spravit: jednoduchy experiment: mozem si simulovat vypadavanie jednotlivych kamier - ze je tam nulova matica tych hodnot v kamere
- upgrade: do DP1 tie augmentacie lepsie
