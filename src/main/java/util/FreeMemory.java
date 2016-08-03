package util;

public class FreeMemory {
    @SuppressWarnings("FieldCanBeLocal")
    private static boolean ON = true;

    public static long get(boolean garbageCollect, int garbageCollectNr) {
        if (!ON)
            return 0;

        if (garbageCollect)
            for (int i = 0; i < garbageCollectNr; i += 1)
                System.gc();

        double maxMem = ((double) Runtime.getRuntime().maxMemory()) / 1024 / 1024;
        double freeMem  = ((double) Runtime.getRuntime().freeMemory()) / 1024 / 1024;
        double totalMem = ((double) Runtime.getRuntime().totalMemory()) / 1024 / 1024;

        double actualMem = maxMem - (totalMem - freeMem);
        return ((long) actualMem);
    }

}
