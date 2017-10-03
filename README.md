This is just a collection of experimental implementations of various AI algorithms... including some basic genetic
algorithms and NEAT (https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)... very much a work in 
progress...

#### Building (currently requires JDK 8 - JDK 9 doesn't work yet):
##### From root:
mvn package

#### Running:
##### jar:
java -cp neatsample/target/jfx/app/neatsample-1.0-SNAPSHOT-jfx.jar matgr.ai.neatsample.App
##### native:
native packages should be placed in neatsample/target/jfx/native

